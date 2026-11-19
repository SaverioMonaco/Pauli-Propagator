//! Compiling a tape: the propagation done once, with box numbers in place of
//! coefficients.
//!
//! The gates are walked in reverse (Heisenberg picture), exactly like the
//! sum-of-terms engine, but a Pauli word carries no terms: only its box
//! number. For every rotation, each word the gate changes is paired with the
//! word it turns into, and the pair is written to the tape as `[a, b]` (see
//! the module doc in `mod.rs` for the order). A Clifford gate only renames
//! words, and records the boxes whose sign flips.
//!
//! With `k1`, a word of Pauli weight above k1 is dropped right after the gate
//! that makes it, as in the sum-of-terms engine. A rotation never changes a
//! word's weight (it swaps two non-identity letters on its qubit), so only a
//! Clifford gate can make a word too heavy; its box is dropped (it leaves the
//! word table, and its value is never read again).
//!
//! A controlled rotation (CRX, CRY, CRZ) is compiled into blocks: its rule
//! table splits the words it changes into groups of 2 or 4 that mix among
//! themselves (one `BlockProg` per group, built from the table), and every
//! such group of existing words becomes a `Block`. Its members can differ in
//! weight, so a member too heavy for k1 is `NONE`. The angle is the half
//! angle, as everywhere in pprop for these gates: a trainable parameter is
//! already θ/2, a fixed angle is halved here.
//!
//! The two exact pruners (dead qubit, XY weight) drop a word that can never
//! reach the readout.
//!
//! With k2, every box also carries the lowest order anything can reach it
//! with (`low`). A word whose lowest order would be above k2 can only hold
//! zeros, so its box is not made, or dropped, as the sum-of-terms engine drops
//! a word once all its terms are cut. A rotation partner that would be born
//! that way is not made at all; its word then keeps only its cos part (a
//! single).
//!
//! With `coeff_threshold`, every box also carries an upper bound on its
//! coefficient over all parameters (`bound`): `|coeff|` for the observable;
//! `√(Ba² + Bb²)` for both words of a trainable rotation pair (|c|, |s| with
//! c² + s² = 1); `|c|·Ba + |s|·Bb` with a fixed angle; `Σ |coeff|·max|sⁿcᵐ|·B`
//! for a block member. Without k2, the observable's norm caps every bound (the
//! propagation is orthogonal and truncations only remove). A word whose bound
//! is below the threshold is dropped, or not made, and its bound is added to
//! `dropped_bound`: without k1 and k2, the result is off by at most that. As in the sum-of-terms engine, a word is checked just
//! before a gate that changes it, against the gates from that one on.

use super::{Angle, Block, BlockProg, Step, Tape, NONE};
use crate::gates::*;
use crate::pauli::*;
use crate::pruning::*;
use rustc_hash::FxHashMap;

/// Every word met so far, and its box.
struct Boxes<const NW: usize> {
    /// box -> its word
    words: Vec<PauliKey<NW>>,
    /// word -> its box
    index: FxHashMap<PauliKey<NW>, u32>,
    /// box -> the last step (numbered from 1) that put it in a pair; 0 = none.
    /// A box already in a pair of the current step is skipped, so a pair whose
    /// two words both exist is written once, not once from each side.
    stamp: Vec<u32>,
    /// box -> false once its word has been dropped (k1, pruners, k2).
    alive: Vec<bool>,
    /// box -> the lowest order (number of factors) it holds, with k2; a new
    /// box starts at `UNREACHED` (it holds nothing yet).
    low: Vec<u32>,
    /// box -> an upper bound on its coefficient over all parameters, with
    /// coeff_threshold; a new box starts at 0.
    bound: Vec<f64>,
}

/// The `low` of a box that holds nothing yet.
const UNREACHED: u32 = u32::MAX;

/// The coefficient threshold, as the compiler sees it.
#[derive(Clone, Copy)]
struct Threshold {
    /// Drop a word whose bound is below this.
    value: f64,
    /// No coefficient can exceed this (the observable's norm, without k2).
    cap: f64,
}

/// The largest `|sin^n cos^m|` over all angles: at sin² = n / (n + m).
fn max_sin_cos(n: u8, m: u8) -> f64 {
    if n == 0 || m == 0 {
        return 1.0;
    }
    let (n, m) = (n as f64, m as f64);
    ((n / (n + m)).powf(n) * (m / (n + m)).powf(m)).sqrt()
}

/// How large one block entry's factor `coeff · sⁿ · cᵐ` can be.
fn entry_size(angle: &Angle, coeff: f64, n: u8, m: u8) -> f64 {
    match *angle {
        Angle::Param(_) => coeff.abs() * max_sin_cos(n, m),
        Angle::Fixed { cos, sin } => (coeff * sin.powi(n as i32) * cos.powi(m as i32)).abs(),
    }
}

/// The k2 truncation, as the compiler sees it: how many orders each step moves
/// a contribution up (see `orders.rs`).
#[derive(Clone, Copy)]
struct Orders {
    k2: u32,
    sine: bool,
}

impl Orders {
    /// (cos part, sin part) of a rotation.
    fn rotation(&self, angle: &Angle) -> (u32, u32) {
        match angle {
            Angle::Param(_) if self.sine => (0, 1),
            Angle::Param(_) => (1, 1),
            Angle::Fixed { .. } => (0, 0),
        }
    }

    /// A block entry with `n_sin` sin and `n_cos` cos factors.
    fn entry(&self, angle: &Angle, n_sin: u8, n_cos: u8) -> u32 {
        match angle {
            Angle::Param(_) if self.sine => n_sin as u32,
            Angle::Param(_) => (n_sin + n_cos) as u32,
            Angle::Fixed { .. } => 0,
        }
    }
}

impl<const NW: usize> Boxes<NW> {
    fn new() -> Self {
        Boxes {
            words: Vec::new(),
            index: FxHashMap::default(),
            stamp: Vec::new(),
            alive: Vec::new(),
            low: Vec::new(),
            bound: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.words.len()
    }

    /// The box of `word`, made now if the word is new. A new word's box starts
    /// at 0 when the tape is evaluated: it did not exist before.
    fn get_or_new(&mut self, word: PauliKey<NW>) -> u32 {
        if let Some(&b) = self.index.get(&word) {
            return b;
        }
        let b = self.words.len() as u32;
        self.words.push(word);
        self.index.insert(word, b);
        self.stamp.push(0);
        self.alive.push(true);
        self.low.push(UNREACHED);
        self.bound.push(0.0);
        b
    }

    /// With a threshold, drop box `b` if its bound is below it, and count it.
    fn cut(&mut self, b: usize, threshold: Option<Threshold>, tape: &mut Tape) {
        if let Some(t) = threshold {
            if self.alive[b] && self.bound[b] < t.value {
                tape.n_dropped += 1;
                tape.dropped_bound += self.bound[b];
                self.drop_box(b);
            }
        }
    }

    /// Drop box `b`: its word leaves the table, so if the word comes back
    /// later it gets a new box starting at 0, and this box is never read again.
    fn drop_box(&mut self, b: usize) {
        self.index.remove(&self.words[b]);
        self.alive[b] = false;
    }

    /// Give boxes new words, and drop boxes, all at once: a Clifford gate is a
    /// one-to-one map of words, so two boxes can swap names (SWAP turns XZ into
    /// ZX and ZX into XZ). All old names are removed first, then the new ones
    /// added.
    fn rename(&mut self, renames: &[(usize, PauliKey<NW>)], drops: &[usize]) {
        for &b in drops {
            self.drop_box(b);
        }
        for &(b, _) in renames {
            self.index.remove(&self.words[b]);
        }
        for &(b, word) in renames {
            self.words[b] = word;
            let previous = self.index.insert(word, b as u32);
            assert!(previous.is_none(), "pprop_rs: a Clifford gate gave two boxes the same word");
        }
    }
}

fn is_clifford(kind: u8) -> bool {
    matches!(kind, H | S | SX | SWAP | CNOT | CY | CZ)
}

/// What a Clifford `gate` makes of `word`: its new word and sign, or `None`
/// if the gate leaves it as it is. Same rule tables as the sum-of-terms
/// engine (`evolve_clifford1q`, `evolve_controlled`, `evolve_swap`).
fn clifford_image<const NW: usize>(gate: &GateSpec<NW>, word: PauliKey<NW>) -> Option<(PauliKey<NW>, i8)> {
    let (x, z) = word;
    let w0 = gate.wire0;
    let one = |rule: fn(u8) -> Option<(u8, i8)>| {
        rule(pauli_label(&x, &z, w0)).map(|(out, sign)| (set_label(x, z, w0, out), sign))
    };
    let two = |rule: fn(u8) -> Option<(u8, u8, i8)>| {
        let w1 = gate.wire1 as u32;
        let code = (pauli_label(&x, &z, w0) << 2) | pauli_label(&x, &z, w1);
        rule(code).map(|(out_c, out_t, sign)| {
            let (nx, nz) = set_label(x, z, w0, out_c);
            (set_label(nx, nz, w1, out_t), sign)
        })
    };
    match gate.kind {
        H => one(h_rule),
        S => one(s_rule),
        SX => one(sx_rule),
        CNOT => two(cnot_rule),
        CY => two(cy_rule),
        CZ => two(cz_rule),
        SWAP => {
            let w1 = gate.wire1 as u32;
            let (l0, l1) = (pauli_label(&x, &z, w0), pauli_label(&x, &z, w1));
            if l0 == l1 {
                return None;
            }
            let (nx, nz) = set_label(x, z, w0, l1);
            Some((set_label(nx, nz, w1, l0), 1))
        }
        _ => unreachable!("not a Clifford gate: kind {}", gate.kind),
    }
}

/// The rule table of a one-qubit rotation, or `None` for a gate the tape
/// cannot compile yet.
///
/// T is a rotation too: its rule (`t_rule`) is X -> X/√2 − Y/√2 and
/// Y -> Y/√2 + X/√2, which is RZ's pairing of X and Y with cos = sin = 1/√2.
/// So it uses RZ's table to find and order its pairs, and a fixed angle.
fn rotation_rule(kind: u8) -> Option<fn(u8) -> Option<(u8, i8)>> {
    match kind {
        RX => Some(rx_rule),
        RY => Some(ry_rule),
        RZ | T => Some(rz_rule),
        _ => None,
    }
}

/// What `compile` returns: the tape, and for debugging the word of every box,
/// whether it is still alive at the end, and its final coefficient bound
/// (with coeff_threshold; 0 otherwise).
pub(crate) type Compiled<const NW: usize> = (Tape, Vec<PauliKey<NW>>, Vec<bool>, Vec<f64>);

/// Compile one observable through `gates` (in circuit order), dropping every
/// word of weight above `k1` (`k1 < 0`: no limit) and, if asked, the words
/// the exact pruners find dead. `k2` (`< 0`: none) and `sine_order` only set
/// how the tape is evaluated (see `orders.rs`).
pub(crate) fn compile<const NW: usize>(
    gates: &[GateSpec<NW>],
    observable: &[(PauliKey<NW>, f64)],
    k1: i64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
    k2: i64,
    sine_order: bool,
    coeff_threshold: f64,
) -> Result<Compiled<NW>, String> {
    let too_heavy = |w: &PauliKey<NW>| k1 >= 0 && combined_weight(&w.0, &w.1) > k1 as u32;
    let orders = if k2 >= 0 { Some(Orders { k2: k2 as u32, sine: sine_order }) } else { None };
    // The gates in propagation order, and what the pruners need of them.
    let gate_masks: Vec<[u64; NW]> = gates.iter().rev().map(|g| g.wire_mask).collect();
    let active_from = dead_qubit_masks(&gate_masks);
    // Can a word reach the readout, seen just before gate `i` (propagation order)?
    let pruned = |w: &PauliKey<NW>, i: usize| {
        (use_dead_qubit_pruner && !mask_subset(&w.0, &active_from[i]))
            || (use_xy_weight_pruner && !xy_reachable(&w.0, &gate_masks[i..]))
    };
    let mut boxes = Boxes::<NW>::new();
    let mut tape = Tape {
        n_boxes: 0,
        init: Vec::new(),
        steps: Vec::new(),
        pairs: Vec::new(),
        negs: Vec::new(),
        singles: Vec::new(),
        blocks: Vec::new(),
        block_progs: Vec::new(),
        readout: Vec::new(),
        k2,
        sine_order,
        n_dropped: 0,
        dropped_bound: 0.0,
    };

    // The observable's words are the first boxes. A word listed twice gets
    // one box with the coefficients added.
    for &(word, coeff) in observable {
        let b = boxes.get_or_new(word);
        boxes.low[b as usize] = 0; // a plain coefficient has no factors
        match tape.init.iter_mut().find(|(ib, _)| *ib == b) {
            Some((_, c)) => *c += coeff,
            None => tape.init.push((b, coeff)),
        }
    }
    let threshold = (coeff_threshold >= 0.0).then(|| {
        let norm = tape.init.iter().map(|(_, c)| c * c).sum::<f64>().sqrt();
        Threshold { value: coeff_threshold, cap: if k2 >= 0 { f64::INFINITY } else { norm } }
    });
    if threshold.is_some() {
        for i in 0..tape.init.len() {
            let (b, c) = tape.init[i];
            boxes.bound[b as usize] = c.abs();
            boxes.cut(b as usize, threshold, &mut tape);
        }
    }
    // The observable's own words are cut to k1 right after the first gate that
    // touches any of them, not before: that gate still sees them whole, as in
    // the sum-of-terms engine (a first CNOT can turn a too-heavy ZZ into IZ).
    let mut obs_support = [0u64; NW];
    for (w, _) in observable {
        mask_or_assign(&mut obs_support, &mask_or(&w.0, &w.1));
    }
    let mut observable_cut = false;
    // The groups of each controlled-rotation kind, built when first needed.
    let mut cr_groups: FxHashMap<u8, CrGroups> = FxHashMap::default();

    for (i, gate) in gates.iter().rev().enumerate() {
        let pruned_here = |w: &PauliKey<NW>| pruned(w, i);
        if is_clifford(gate.kind) {
            // Rename every word the gate changes; remember the minus signs;
            // drop the words it makes too heavy.
            let first = tape.negs.len() as u32;
            let mut renames = Vec::new();
            let mut drops = Vec::new();
            for b in 0..boxes.len() {
                if !boxes.alive[b] {
                    continue;
                }
                if let Some((word, sign)) = clifford_image(gate, boxes.words[b]) {
                    if pruned_here(&boxes.words[b]) || too_heavy(&word) {
                        drops.push(b);
                        continue;
                    }
                    if sign < 0 {
                        tape.negs.push(b as u32);
                    }
                    if word != boxes.words[b] {
                        renames.push((b, word));
                    }
                }
            }
            boxes.rename(&renames, &drops);
            let last = tape.negs.len() as u32;
            if last > first {
                tape.steps.push(Step::Negate { first, last });
            }
        } else if is_controlled_rotation(gate.kind) {
            let groups = cr_groups.entry(gate.kind).or_insert_with(|| CrGroups::new(gate.kind, &mut tape));
            controlled_rotation_step(gate, groups, &mut boxes, &mut tape, &too_heavy, &pruned_here, orders, threshold);
        } else {
            rotation_step(gate, &mut boxes, &mut tape, &too_heavy, &pruned_here, orders, threshold)?;
        }

        if !observable_cut && mask_intersects(&gate.wire_mask, &obs_support) {
            observable_cut = true;
            for b in 0..boxes.len() {
                if boxes.alive[b] && too_heavy(&boxes.words[b]) {
                    boxes.drop_box(b);
                }
            }
        }
    }

    tape.n_boxes = boxes.len() as u32;
    tape.readout = (0..boxes.len() as u32)
        .filter(|&b| boxes.alive[b as usize] && mask_is_zero(&boxes.words[b as usize].0))
        .collect();
    Ok((tape, boxes.words, boxes.alive, boxes.bound))
}

/// One rotation gate: pair every word it changes with the word it turns into.
/// A word the pruners reject is dropped instead.
fn rotation_step<const NW: usize>(
    gate: &GateSpec<NW>,
    boxes: &mut Boxes<NW>,
    tape: &mut Tape,
    too_heavy: &impl Fn(&PauliKey<NW>) -> bool,
    pruned: &impl Fn(&PauliKey<NW>) -> bool,
    orders: Option<Orders>,
    threshold: Option<Threshold>,
) -> Result<(), String> {
    let rule = rotation_rule(gate.kind)
        .ok_or_else(|| format!("the tape cannot compile gate kind {} yet", gate.kind))?;
    // A trainable angle is read at evaluation time; a fixed one is turned into
    // its cos and sin now, with the same snapping to exact zeros at quarter
    // turns as the sum-of-terms engine (`fixed_sin_cos`).
    let angle = if gate.kind == T {
        // The same constant as `t_rule`, not sin(π/4), which differs from it
        // in the last digit.
        Angle::Fixed { cos: std::f64::consts::FRAC_1_SQRT_2, sin: std::f64::consts::FRAC_1_SQRT_2 }
    } else if gate.param >= 0 {
        Angle::Param(gate.param as u32)
    } else if let Some(theta) = gate.fixed {
        let (sin, cos) = fixed_sin_cos(theta);
        Angle::Fixed { cos, sin }
    } else {
        return Err(format!("rotation (gate kind {}) has neither a parameter nor an angle", gate.kind));
    };
    let step = tape.steps.len() as u32 + 1;
    let first = tape.pairs.len() as u32;
    let sfirst = tape.singles.len() as u32;

    // Only the words that exist before this gate can be changed by it; the
    // partners it creates are already in a pair of this step.
    for b in 0..boxes.len() {
        if !boxes.alive[b] || boxes.stamp[b] == step {
            continue;
        }
        // A too-heavy word can only be one of the observable's, before its
        // cut: its partner has the same weight, so both would be dropped.
        if too_heavy(&boxes.words[b]) {
            continue;
        }
        let (x, z) = boxes.words[b];
        let Some((out, sign)) = rule(pauli_label(&x, &z, gate.wire0)) else {
            continue; // the gate commutes with this word
        };
        if pruned(&boxes.words[b]) {
            boxes.drop_box(b);
            continue;
        }
        // An existing partner is not checked: if the pruners reject it, its
        // whole future, including what it gives this word here, adds 0 to the
        // readout (they are exact), so keeping its box changes no value and
        // saves making a fresh one.
        let partner_word = set_label(x, z, gate.wire0, out);
        let existing = boxes.index.get(&partner_word).copied();
        if existing.is_none() {
            // A new partner would only get this word's sin part: at orders
            // low + ss and up (k2), and at most |sin|·bound in size (threshold).
            // If that is above k2 or below the threshold it is not made, and the
            // word keeps its cos part, as a single.
            let lw = boxes.low[b];
            let k2_doomed = orders.is_some_and(|o| lw.saturating_add(o.rotation(&angle).1) > o.k2);
            let (cos_size, sin_size) = match angle {
                Angle::Param(_) => (1.0, 1.0),
                Angle::Fixed { cos, sin } => (cos.abs(), sin.abs()),
            };
            let partner_bound = sin_size * boxes.bound[b];
            let small = threshold.is_some_and(|t| partner_bound < t.value);
            if k2_doomed || small {
                if small && !k2_doomed {
                    tape.n_dropped += 1; // the partner's part is lost
                    tape.dropped_bound += partner_bound;
                }
                boxes.stamp[b] = step;
                if let Some(o) = orders {
                    let l = lw.saturating_add(o.rotation(&angle).0);
                    if l > o.k2 {
                        boxes.drop_box(b);
                        continue;
                    }
                    boxes.low[b] = l;
                }
                if threshold.is_some() {
                    boxes.bound[b] *= cos_size;
                    boxes.cut(b, threshold, tape);
                    if !boxes.alive[b] {
                        continue;
                    }
                }
                tape.singles.push(b as u32);
                continue;
            }
        }
        let partner = existing.unwrap_or_else(|| boxes.get_or_new(partner_word));
        // `word -> cos·word + sign·sin·partner`: with sign +1 the word is
        // `a`; with sign -1 the partner is (the rule is antisymmetric).
        let pair = if sign > 0 { [b as u32, partner] } else { [partner, b as u32] };
        boxes.stamp[pair[0] as usize] = step;
        boxes.stamp[pair[1] as usize] = step;
        tape.pairs.push(pair);
        if let Some(o) = orders {
            // new a = c·a + (−s)·b, new b = s·a + c·b: each part moves up by its factors.
            let (sc, ss) = o.rotation(&angle);
            let (a, bb) = (pair[0] as usize, pair[1] as usize);
            let (la, lb) = (boxes.low[a], boxes.low[bb]);
            boxes.low[a] = la.saturating_add(sc).min(lb.saturating_add(ss));
            boxes.low[bb] = la.saturating_add(ss).min(lb.saturating_add(sc));
            for m in [a, bb] {
                if boxes.low[m] > o.k2 {
                    boxes.drop_box(m); // it now holds only orders above k2, i.e. zeros
                }
            }
        }
        if let Some(t) = threshold {
            let (a, bb) = (pair[0] as usize, pair[1] as usize);
            let (ba, bb_) = (boxes.bound[a], boxes.bound[bb]);
            let (na, nb) = match angle {
                Angle::Param(_) => {
                    let r = (ba * ba + bb_ * bb_).sqrt();
                    (r, r)
                }
                Angle::Fixed { cos, sin } => {
                    (cos.abs() * ba + sin.abs() * bb_, sin.abs() * ba + cos.abs() * bb_)
                }
            };
            boxes.bound[a] = na.min(t.cap);
            boxes.bound[bb] = nb.min(t.cap);
            boxes.cut(a, threshold, tape);
            boxes.cut(bb, threshold, tape);
        }
    }

    let last = tape.pairs.len() as u32;
    let slast = tape.singles.len() as u32;
    if last > first || slast > sfirst {
        tape.steps.push(Step::Rotate { angle, first, last, sfirst, slast });
    }
    Ok(())
}

fn is_controlled_rotation(kind: u8) -> bool {
    matches!(kind, CRX | CRY | CRZ)
}

fn cr_rule_of(kind: u8) -> fn(u8) -> Option<CRRule> {
    match kind {
        CRX => crx_rule,
        CRY => cry_rule,
        CRZ => crz_rule,
        _ => unreachable!("not a controlled rotation: kind {}", kind),
    }
}

/// The groups of one controlled-rotation kind. A word's two letters on the
/// gate's (control, target) give its code `control << 2 | target`; the codes
/// the rule table changes split into groups that only mix among themselves
/// (CRX: one group of 4 with control I/Z, four pairs with control X/Y).
struct CrGroups {
    /// Per group: the codes of its members, in member order, and its prog.
    groups: Vec<(Vec<u8>, u32)>,
    /// Per code: (its group, its member position), or None if unchanged.
    of_code: [Option<(usize, usize)>; 16],
}

impl CrGroups {
    /// Find the groups of `kind`'s rule table and add one `BlockProg` per
    /// group to `tape`. An entry `code -> coeff·sⁿ·cᵐ·out` of the table becomes
    /// the prog entry `(out member, code member, coeff, n, m)`.
    fn new(kind: u8, tape: &mut Tape) -> Self {
        let rule = cr_rule_of(kind);
        let mut of_code: [Option<(usize, usize)>; 16] = [None; 16];
        let mut groups: Vec<(Vec<u8>, u32)> = Vec::new();
        for start in 0..16u8 {
            if rule(start).is_none() || of_code[start as usize].is_some() {
                continue;
            }
            // Everything reachable from `start` through the table is one group.
            let mut members = vec![start];
            let mut i = 0;
            while i < members.len() {
                for &(oc, ot, ..) in rule(members[i]).unwrap() {
                    let out = (oc << 2) | ot;
                    if !members.contains(&out) {
                        members.push(out);
                    }
                }
                i += 1;
            }
            members.sort_unstable();
            assert!(members.len() <= 4, "pprop_rs: a controlled-rotation group has more than 4 words");
            let g = groups.len();
            let mut entries = Vec::new();
            for (pos, &code) in members.iter().enumerate() {
                of_code[code as usize] = Some((g, pos));
                for &(oc, ot, coeff, n_sin, n_cos) in rule(code).unwrap() {
                    let out = members.iter().position(|&m| m == (oc << 2) | ot).unwrap();
                    entries.push((out as u8, pos as u8, coeff, n_sin, n_cos));
                }
            }
            let prog = tape.block_progs.len() as u32;
            tape.block_progs.push(BlockProg { size: members.len() as u8, entries });
            groups.push((members, prog));
        }
        CrGroups { groups, of_code }
    }
}

/// One controlled rotation: every group of words it changes becomes a block.
/// A visited word the pruners reject is dropped; the other members are not
/// checked (as for rotations: what a rejected word contributes adds 0 to the
/// readout anyway). A member that does not exist yet gets a new box at 0, or
/// `NONE` if it is too heavy for k1.
fn controlled_rotation_step<const NW: usize>(
    gate: &GateSpec<NW>,
    groups: &CrGroups,
    boxes: &mut Boxes<NW>,
    tape: &mut Tape,
    too_heavy: &impl Fn(&PauliKey<NW>) -> bool,
    pruned: &impl Fn(&PauliKey<NW>) -> bool,
    orders: Option<Orders>,
    threshold: Option<Threshold>,
) {
    let (control, target) = (gate.wire0, gate.wire1 as u32);
    let angle = if gate.param >= 0 {
        Angle::Param(gate.param as u32)
    } else {
        // Fixed angle: the rule tables are in θ/2.
        let (sin, cos) = fixed_sin_cos(gate.fixed.expect("a controlled rotation has a parameter or an angle") / 2.0);
        Angle::Fixed { cos, sin }
    };
    let step = tape.steps.len() as u32 + 1;
    let first = tape.blocks.len() as u32;

    for b in 0..boxes.len() {
        if !boxes.alive[b] || boxes.stamp[b] == step {
            continue;
        }
        let (x, z) = boxes.words[b];
        let code = (pauli_label(&x, &z, control) << 2) | pauli_label(&x, &z, target);
        let Some((g, _)) = groups.of_code[code as usize] else {
            continue; // the gate leaves this word as it is
        };
        if pruned(&boxes.words[b]) {
            boxes.drop_box(b);
            continue;
        }
        let (codes, prog) = &groups.groups[g];
        let words: Vec<PauliKey<NW>> = codes
            .iter()
            .map(|&c| {
                let (mx, mz) = set_label(x, z, control, c >> 2);
                set_label(mx, mz, target, c & 3)
            })
            .collect();
        let existing: Vec<Option<u32>> = words.iter().map(|w| boxes.index.get(w).copied()).collect();
        // With k2: the lowest order each member will hold after the gate, from
        // the members that hold something now.
        let mut new_low = [UNREACHED; 4];
        if let Some(o) = orders {
            for &(out, inp, _, ns, nc) in &tape.block_progs[*prog as usize].entries {
                if let Some(e) = existing[inp as usize] {
                    let l = boxes.low[e as usize].saturating_add(o.entry(&angle, ns, nc));
                    new_low[out as usize] = new_low[out as usize].min(l);
                }
            }
        }
        let doomed = |m: usize| orders.is_some_and(|o| new_low[m] > o.k2);
        // With a threshold: how large each member can be after the gate.
        let mut new_bound = [0.0f64; 4];
        if let Some(t) = threshold {
            for &(out, inp, k, ns, nc) in &tape.block_progs[*prog as usize].entries {
                if let Some(e) = existing[inp as usize] {
                    new_bound[out as usize] += entry_size(&angle, k, ns, nc) * boxes.bound[e as usize];
                }
            }
            new_bound.iter_mut().for_each(|b| *b = b.min(t.cap));
        }
        let small = |m: usize| threshold.is_some_and(|t| new_bound[m] < t.value);
        let mut members = [NONE; 4];
        for (m, word) in words.iter().enumerate() {
            members[m] = match existing[m] {
                Some(e) => e,
                None if too_heavy(word) || doomed(m) => NONE,
                None if small(m) => {
                    tape.n_dropped += 1; // what this member would have held is lost
                    tape.dropped_bound += new_bound[m];
                    NONE
                }
                None => boxes.get_or_new(*word),
            };
            if members[m] != NONE {
                boxes.stamp[members[m] as usize] = step;
            }
        }
        tape.blocks.push(Block { prog: *prog, members });
        if orders.is_some() {
            for m in 0..codes.len() {
                if members[m] != NONE {
                    boxes.low[members[m] as usize] = new_low[m];
                    if doomed(m) {
                        boxes.drop_box(members[m] as usize);
                    }
                }
            }
        }
        if threshold.is_some() {
            for m in 0..codes.len() {
                if members[m] != NONE {
                    boxes.bound[members[m] as usize] = new_bound[m];
                    boxes.cut(members[m] as usize, threshold, tape);
                }
            }
        }
    }

    let last = tape.blocks.len() as u32;
    if last > first {
        tape.steps.push(Step::Block { angle, first, last });
    }
}
