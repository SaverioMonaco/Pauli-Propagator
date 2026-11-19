//! The sum-of-terms engine: every Pauli path is kept as its own term.

mod arena;
mod evaluator;
mod evolve;

pub(crate) use arena::*;
pub(crate) use evaluator::*;
pub(crate) use evolve::*;

use crate::gates::*;
use crate::pauli::*;
use crate::pruning::*;
use crate::threads::*;
use pyo3::prelude::*;
use rustc_hash::FxHashMap;

// One trigonometric product term as Python sees it: (coeff, sin_indices,
// cos_indices) means coeff * prod(sin(theta_i) for i in sin_indices) *
// prod(cos(theta_j) for j in cos_indices), e.g. (0.5, [0], [1]) =
// 0.5*sin(theta_0)*cos(theta_1), and (-0.25, [], [0, 0]) =
// -0.25*cos(theta_0)^2 (repeated index = squared). This is only the
// boundary format; inside the propagation loop a term is a `Term`.
pub(crate) type CoeffTerm = (f64, Vec<u32>, Vec<u32>);
// Pauli word -> its current coefficient, for the words one thread owns.
pub(crate) type WordMap<const NW: usize> = FxHashMap<PauliKey<NW>, Vec<Term>>;



/// Merge `terms` into `key`'s coefficient. A new word takes ownership of the
/// vector as-is; only a word that already exists pays for a copy. Returns
/// whether the word already existed.
pub(crate) fn insert_or_extend<const NW: usize>(
    map: &mut FxHashMap<PauliKey<NW>, Vec<Term>>,
    key: PauliKey<NW>,
    mut terms: Vec<Term>,
) -> bool {
    use std::collections::hash_map::Entry;
    match map.entry(key) {
        Entry::Vacant(slot) => {
            slot.insert(terms);
            false
        }
        Entry::Occupied(mut slot) => {
            let existing = slot.get_mut();
            if existing.len() < terms.len() {
                std::mem::swap(existing, &mut terms);
            }
            existing.append(&mut terms);
            true
        }
    }
}

/// Sum a word's terms into one, for a propagation in which every term is a
/// plain number (`mono == ROOT`). The terms are summed in order of value, so
/// the result does not depend on the order they were appended in, which
/// changes with the number of threads. Returns `false` when the sum is
/// exactly zero and the word should be dropped.
pub(crate) fn sum_scalar_terms(terms: &mut Vec<Term>) -> bool {
    if terms.len() > 1 {
        terms.sort_unstable_by(|a, b| a.coeff.total_cmp(&b.coeff));
        let sum: f64 = terms.iter().map(|t| t.coeff).sum();
        terms.truncate(1);
        terms[0].coeff = sum;
    }
    terms.first().is_some_and(|t| t.coeff != 0.0)
}


/// Order of the terms in a finished expression: by index lists, then by
/// coefficient.
pub(crate) fn cmp_terms(a: &CoeffTerm, b: &CoeffTerm) -> std::cmp::Ordering {
    (&a.1, &a.2)
        .cmp(&(&b.1, &b.2))
        .then_with(|| a.0.total_cmp(&b.0))
}

/// The terms of every word with no X/Y, i.e. with a nonzero expectation in
/// |0...0>, sorted by `cmp_terms`. Iterating the maps gives an order that
/// depends on hashing and on how the words were split over threads; sorting
/// makes the expression the same either way.
pub(crate) fn to_expectation<const NW: usize>(
    shards: &[WordMap<NW>],
    arenas: &[Monomials],
    parallel: bool,
) -> Vec<CoeffTerm> {
    let runs = run_all(shards.iter().collect(), parallel, |shard: &WordMap<NW>| {
        let mut run = Vec::new();
        for ((x, _z), terms) in shard.iter() {
            if mask_is_zero(x) {
                for t in terms {
                    let (s, cc) = to_lists(arenas, t);
                    run.push((t.coeff, s, cc));
                }
            }
        }
        run.sort_unstable_by(cmp_terms);
        run
    });
    let mut out: Vec<CoeffTerm> = runs.into_iter().flatten().collect();
    // A stable sort finds the per-shard runs and just merges them.
    out.sort_by(cmp_terms);
    out
}


// ---------------------------------------------------------------------
// Heisenberg propagation loop
// ---------------------------------------------------------------------

/// Apply the k1/k2/coefficient truncations to one word; `false` means the
/// word has nothing left and should be dropped.
pub(crate) fn truncate<const NW: usize>(
    key: &PauliKey<NW>,
    terms: &mut Vec<Term>,
    k1: i64,
    k2: i64,
    coeff_threshold: f64,
) -> bool {
    if k1 >= 0 && combined_weight(&key.0, &key.1) > k1 as u32 {
        return false;
    }
    if k2 >= 0 {
        let k2u = k2 as u32;
        terms.retain(|t| t.depth <= k2u);
    }
    if coeff_threshold >= 0.0 {
        terms.retain(|t| t.coeff.abs() >= coeff_threshold);
    }
    !terms.is_empty()
}

/// A run of consecutive gates (in propagation order) on pairwise disjoint
/// wires. Such gates commute, and a word's labels on one gate's wires are
/// untouched by the others, so the gates a word will meet in the layer can
/// be read off the word once, before any of them is applied.
pub(crate) struct Layer<const NW: usize> {
    start: usize,
    end: usize,
    mask: [u64; NW],
}

/// Split `gate_masks` greedily into maximal runs of wire-disjoint gates.
pub(crate) fn build_layers<const NW: usize>(gate_masks: &[[u64; NW]]) -> Vec<Layer<NW>> {
    let mut layers: Vec<Layer<NW>> = Vec::new();
    for (i, gm) in gate_masks.iter().enumerate() {
        match layers.last_mut() {
            Some(layer) if !mask_intersects(&layer.mask, gm) => {
                layer.end = i + 1;
                mask_or_assign(&mut layer.mask, gm);
            }
            _ => layers.push(Layer {
                start: i,
                end: i + 1,
                mask: *gm,
            }),
        }
    }
    layers
}

/// Everything about the circuit and the truncation settings that stays fixed
/// for a whole propagation, shared read-only by every thread.
pub(crate) struct Circuit<'a, const NW: usize> {
    reversed: &'a [GateSpec<NW>],
    gate_masks: &'a [[u64; NW]],
    active_qubits_from: &'a [[u64; NW]],
    k1: i64,
    k2: i64,
    coeff_threshold: f64, // < 0 disables
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
}

/// One layer's pass, as seen by each shard's thread.
pub(crate) struct LayerPass<'a, const NW: usize> {
    circuit: &'a Circuit<'a, NW>,
    layer: &'a Layer<NW>,
    // Gate owning each qubit in this layer (usize::MAX = none).
    gate_on_qubit: &'a [usize],
    // Whether the map still holds the observable's untruncated words.
    fresh: bool,
    first_gate: Option<usize>,
    n_shards: usize,
}

// Below this many words a layer runs on the calling thread; spawning
// would cost more than it saves.
pub(crate) const PARALLEL_MIN_WORDS: usize = 1 << 13;

/// First half of a layer, for one shard: move out the words the layer acts
/// on, carry each through its gates in circuit order, and batch the results
/// by the shard that owns them. Also returns the support of the words that
/// stayed.
pub(crate) fn evolve_shard<const NW: usize>(
    pass: &LayerPass<NW>,
    shard: &mut WordMap<NW>,
    mono: &mut Monomials,
) -> (Vec<Emitted<NW>>, [u64; NW]) {
    let c = pass.circuit;
    let truncate = |key: &PauliKey<NW>, terms: &mut Vec<Term>| {
        truncate(key, terms, c.k1, c.k2, c.coeff_threshold)
    };

    // Each moved-out word comes with the (ascending) indices of the gates
    // that act on it.
    let mut acted: Vec<(PauliKey<NW>, Vec<Term>, Vec<usize>)> = Vec::new();
    let mut stayed_active = [0u64; NW];
    shard.retain(|key, terms| {
        let (x, z) = key;
        let support = mask_or(x, z);
        let mut gates: Vec<usize> = Vec::new();
        if mask_intersects(&support, &pass.layer.mask) {
            for_each_bit(&support, |q| {
                let g = pass.gate_on_qubit[q];
                if g != usize::MAX && !gates.contains(&g) && gate_acts(&c.reversed[g], x, z) {
                    gates.push(g);
                }
            });
        }
        if !gates.is_empty() {
            gates.sort_unstable();
            acted.push((*key, std::mem::take(terms), gates));
            return false;
        }
        if pass.fresh && !truncate(key, terms) {
            return false;
        }
        mask_or_assign(&mut stayed_active, &support);
        true
    });

    let mut outgoing: Vec<Emitted<NW>> = (0..pass.n_shards).map(|_| Vec::new()).collect();
    let mut work: Vec<(PauliKey<NW>, Vec<Term>, usize)> = Vec::new();
    let mut emitted: Emitted<NW> = Vec::new();
    for (key, terms, gates) in acted {
        work.push((key, terms, 0));
        while let Some((key, mut terms, pos)) = work.pop() {
            if pos == gates.len() {
                outgoing[shard_of(&key, pass.n_shards)].push((key, terms));
                continue;
            }
            let g = gates[pos];
            if pass.fresh && pos == 0 && pass.first_gate != Some(g) {
                // An observable word the first gate step left alone: it
                // was truncated along with everything else after it.
                if !truncate(&key, &mut terms) {
                    continue;
                }
            }
            // DeadQubitPruner: word is dead if it has X/Y (x-bit set) on a
            // qubit that no remaining gate (incl. this one) touches again.
            if c.use_dead_qubit_pruner && !mask_subset(&key.0, &c.active_qubits_from[g]) {
                continue;
            }
            if c.use_xy_weight_pruner && !xy_reachable(&key.0, &c.gate_masks[g..]) {
                continue;
            }
            evolve_one_gate(&c.reversed[g], key.0, key.1, terms, mono, &mut emitted);
            for (out_key, mut out_terms) in emitted.drain(..) {
                if truncate(&out_key, &mut out_terms) {
                    work.push((out_key, out_terms, pos + 1));
                }
            }
        }
    }
    (outgoing, stayed_active)
}

/// Second half of a layer, for one shard: merge in the words every shard
/// produced for it. With `scalar`, every word reached more than once ends
/// up with one summed term. Returns their support.
pub(crate) fn merge_shard<const NW: usize>(
    shard: &mut WordMap<NW>,
    incoming: Vec<Emitted<NW>>,
    scalar: bool,
) -> [u64; NW] {
    let mut active = [0u64; NW];
    let mut extended: Vec<PauliKey<NW>> = Vec::new();
    for batch in incoming {
        for (key, terms) in batch {
            mask_or_assign(&mut active, &key.0);
            mask_or_assign(&mut active, &key.1);
            if insert_or_extend(shard, key, terms) && scalar {
                extended.push(key);
            }
        }
    }
    // A key can be listed more than once; after its first visit it holds a
    // single term (or is gone), so later visits change nothing.
    for key in extended {
        if let Some(terms) = shard.get_mut(&key) {
            if !sum_scalar_terms(terms) {
                shard.remove(&key);
            }
        }
    }
    active
}

/// Propagate one observable backwards through the circuit.
///
/// The words are split over `shards` by `shard_of`, one shard per thread,
/// and every thread appends to its own monomial arena. Each layer (see
/// `Layer`) is one pass: every shard moves out the words some gate in the
/// layer acts on, leaving the rest where they are; each moved-out word is
/// carried through the gates that act on it, in circuit order; and the
/// results are merged into whichever shard owns them. The result is the same
/// as applying one gate at a time to the whole map, with pruning and
/// truncation after every gate:
///
/// * Truncation (k1/k2/coefficient) depends only on a word's key and terms,
///   so it only has to run when a gate produces a word. The observable's own
///   words are truncated once, like everything else after the first gate
///   step that touches any of them.
/// * Both exact pruners are monotone along the circuit: a word rejected at
///   one gate would be rejected at every later gate too (DeadQubitPruner's
///   set of still-reachable qubits only shrinks; XYWeightPruner's budget
///   only shrinks as gates are used up). So a word is checked only when a
///   gate is about to act on it, which rejects it exactly when some earlier
///   check would have. A dead word that no gate touches again just sits in
///   the map; it has an X/Y somewhere, so it never reaches the expectation.
///
/// With `scalar` (no trainable gate and no sin/cos factor in the observable,
/// so every term stays `mono == ROOT`), the terms a word collects are summed
/// into one after every layer: the work then scales with the number of
/// distinct words, not of paths. Without it no term is ever compared, so a
/// circuit with trainable parameters pays nothing for this.
pub(crate) fn heisenberg_one<const NW: usize>(
    circuit: &Circuit<NW>,
    layers: &[Layer<NW>],
    mut shards: Vec<WordMap<NW>>,
    arenas: &mut [Monomials],
    scalar: bool,
) -> Vec<CoeffTerm> {
    let n_shards = shards.len();
    let mut active_mask = [0u64; NW];
    for (x, z) in shards.iter().flat_map(|s| s.keys()) {
        mask_or_assign(&mut active_mask, x);
        mask_or_assign(&mut active_mask, z);
    }
    // The first gate the one-gate-at-a-time loop would not skip. Until it
    // has been applied, the observable's own words are untruncated.
    let first_gate = circuit
        .gate_masks
        .iter()
        .position(|gm| mask_intersects(gm, &active_mask));

    let mut gate_on_qubit = vec![usize::MAX; NW * 64];
    let mut compact_at = MIN_COMPACT_NODES;
    // Test-only hook: compact after every layer.
    let compact_every_layer = std::env::var_os("PPROP_RS_COMPACT_EVERY_LAYER").is_some();
    let mut started = false;

    for layer in layers {
        if !mask_intersects(&layer.mask, &active_mask) {
            continue;
        }
        // Only the first layer processed still holds untruncated words.
        let fresh = !started;
        started = true;
        let layer_masks = &circuit.gate_masks[layer.start..layer.end];
        for (g, gm) in (layer.start..).zip(layer_masks) {
            for_each_bit(gm, |q| gate_on_qubit[q] = g);
        }

        let n_words: usize = shards.iter().map(|s| s.len()).sum();
        let parallel = n_shards > 1 && n_words >= PARALLEL_MIN_WORDS;
        let pass = LayerPass {
            circuit,
            layer,
            gate_on_qubit: &gate_on_qubit,
            fresh,
            first_gate,
            n_shards,
        };
        let evolved = run_all(
            shards.iter_mut().zip(arenas.iter_mut()).collect(),
            parallel,
            |(shard, mono)| evolve_shard(&pass, shard, mono),
        );
        let mut next_active = [0u64; NW];
        let mut incoming: Vec<Vec<Emitted<NW>>> = (0..n_shards).map(|_| Vec::new()).collect();
        for (outgoing, stayed_active) in evolved {
            mask_or_assign(&mut next_active, &stayed_active);
            for (dest, batch) in outgoing.into_iter().enumerate() {
                incoming[dest].push(batch);
            }
        }
        let merged = run_all(
            shards.iter_mut().zip(incoming).collect(),
            parallel,
            |(shard, batches)| merge_shard(shard, batches, scalar),
        );
        for mask in &merged {
            mask_or_assign(&mut next_active, mask);
        }
        active_mask = next_active;

        for gm in layer_masks {
            for_each_bit(gm, |q| gate_on_qubit[q] = usize::MAX);
        }
        let n_nodes: usize = arenas.iter().map(|a| a.len()).sum();
        if compact_every_layer
            || n_nodes >= compact_at
            || arenas.iter().any(|a| a.len() >= ARENA_CAPACITY / 2)
        {
            compact(arenas, &mut shards, n_shards > 1);
            let n_nodes: usize = arenas.iter().map(|a| a.len()).sum();
            compact_at = MIN_COMPACT_NODES.max(4 * n_nodes);
        }
    }

    to_expectation(&shards, arenas, n_shards > 1)
}


#[allow(clippy::too_many_arguments)]
pub(crate) fn propagate_batch_impl<const NW: usize>(
    gate_kind: Vec<u8>,
    gate_wire0: Vec<u32>,
    gate_wire1: Vec<i64>,
    gate_param: Vec<i64>,
    gate_fixed: Vec<Option<f64>>,
    k1: i64,
    k2: i64,
    coeff_threshold: f64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
    paulidicts: Vec<Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>>,
    n_threads: usize,
    sine_order: bool,
) -> PyResult<Vec<Vec<CoeffTerm>>> {
    let n_gates = gate_kind.len();
    let mut gates: Vec<GateSpec<NW>> = Vec::with_capacity(n_gates);
    for i in 0..n_gates {
        let mut wire_mask = [0u64; NW];
        set_wire_bit(&mut wire_mask, gate_wire0[i]);
        if gate_wire1[i] >= 0 {
            set_wire_bit(&mut wire_mask, gate_wire1[i] as u32);
        }
        gates.push(GateSpec {
            kind: gate_kind[i],
            wire0: gate_wire0[i],
            wire1: gate_wire1[i],
            param: gate_param[i],
            fixed: gate_fixed[i],
            wire_mask,
        });
    }
    // Heisenberg picture: consume gates in reverse circuit order.
    let reversed: Vec<GateSpec<NW>> = gates.into_iter().rev().collect();
    let n = reversed.len();
    let gate_masks: Vec<[u64; NW]> = reversed.iter().map(|g| g.wire_mask).collect();
    let layers = build_layers(&gate_masks);

    // Suffix union of wire masks: active_qubits_from[i] = qubits touched by
    // gates i..n-1 (used by DeadQubitPruner).
    let mut active_qubits_from = vec![[0u64; NW]; n + 1];
    for i in (0..n).rev() {
        active_qubits_from[i] = mask_or(&active_qubits_from[i + 1], &reversed[i].wire_mask);
    }

    let circuit = Circuit {
        reversed: &reversed,
        gate_masks: &gate_masks,
        active_qubits_from: &active_qubits_from,
        k1,
        k2,
        coeff_threshold,
        use_dead_qubit_pruner,
        use_xy_weight_pruner,
    };
    let n_shards = n_threads.clamp(1, MAX_ARENAS);
    // Only trainable gates push sin/cos factors (fixed ones fold theirs
    // into the coefficient), so without them every term stays a number.
    let all_fixed = reversed.iter().all(|g| g.param < 0);

    let mut results = Vec::with_capacity(paulidicts.len());
    for pd_spec in paulidicts {
        let scalar = all_fixed
            && pd_spec.iter().all(|(_, _, _, s, c)| s.is_empty() && c.is_empty());
        let mut arenas: Vec<Monomials> = (0..n_shards)
            .map(|a| Monomials::new(a, sine_order))
            .collect();
        let mut shards: Vec<WordMap<NW>> = (0..n_shards).map(|_| WordMap::default()).collect();
        for (xw, zw, coeff, sin_idx, cos_idx) in pd_spec {
            let key = (words_to_array::<NW>(&xw), words_to_array::<NW>(&zw));
            let term = arenas[0].term_from_lists(coeff, &sin_idx, &cos_idx);
            shards[shard_of(&key, n_shards)]
                .entry(key)
                .or_default()
                .push(term);
        }
        if scalar {
            for shard in shards.iter_mut() {
                shard.retain(|_, terms| sum_scalar_terms(terms));
            }
        }
        results.push(heisenberg_one(&circuit, &layers, shards, &mut arenas, scalar));
    }

    Ok(results)
}

pub(crate) fn evolve_single_gate_debug_impl<const NW: usize>(
    kind: u8,
    wire0: u32,
    wire1: i64,
    param: i64,
    fixed: Option<f64>,
    x: &[u64],
    z: &[u64],
    terms: Vec<(f64, Vec<u32>, Vec<u32>)>,
) -> Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)> {
    let mut wire_mask = [0u64; NW];
    set_wire_bit(&mut wire_mask, wire0);
    if wire1 >= 0 {
        set_wire_bit(&mut wire_mask, wire1 as u32);
    }
    let gate = GateSpec::<NW> { kind, wire0, wire1, param, fixed, wire_mask };
    let mut mono = Monomials::new(0, false);
    let terms: Vec<Term> = terms.iter().map(|(c, s, cc)| mono.term_from_lists(*c, s, cc)).collect();
    let mut out: Emitted<NW> = Vec::new();
    evolve_one_gate(&gate, words_to_array::<NW>(x), words_to_array::<NW>(z), terms, &mut mono, &mut out);
    let mut rows = Vec::new();
    for ((ox, oz), terms) in out {
        for t in terms {
            let (s, cc) = to_lists(std::slice::from_ref(&mono), &t);
            rows.push((ox.to_vec(), oz.to_vec(), t.coeff, s, cc));
        }
    }
    rows
}
