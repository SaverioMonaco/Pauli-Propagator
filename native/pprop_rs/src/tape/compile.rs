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
//! The two exact pruners (dead qubit, XY weight) drop a word that can never
//! reach the readout. As in the sum-of-terms engine, a word is checked just
//! before a gate that changes it, against the gates from that one on.

use super::{Angle, Step, Tape};
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
    /// box -> false once its word has been dropped (k1).
    alive: Vec<bool>,
}

impl<const NW: usize> Boxes<NW> {
    fn new() -> Self {
        Boxes { words: Vec::new(), index: FxHashMap::default(), stamp: Vec::new(), alive: Vec::new() }
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
        b
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

/// What `compile` returns: the tape, and for debugging the word of every box
/// and whether it is still alive at the end.
pub(crate) type Compiled<const NW: usize> = (Tape, Vec<PauliKey<NW>>, Vec<bool>);

/// Compile one observable through `gates` (in circuit order), dropping every
/// word of weight above `k1` (`k1 < 0`: no limit) and, if asked, the words
/// the exact pruners find dead.
pub(crate) fn compile<const NW: usize>(
    gates: &[GateSpec<NW>],
    observable: &[(PauliKey<NW>, f64)],
    k1: i64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
) -> Result<Compiled<NW>, String> {
    let too_heavy = |w: &PauliKey<NW>| k1 >= 0 && combined_weight(&w.0, &w.1) > k1 as u32;
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
        readout: Vec::new(),
    };

    // The observable's words are the first boxes. A word listed twice gets
    // one box with the coefficients added.
    for &(word, coeff) in observable {
        let b = boxes.get_or_new(word);
        match tape.init.iter_mut().find(|(ib, _)| *ib == b) {
            Some((_, c)) => *c += coeff,
            None => tape.init.push((b, coeff)),
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
        } else {
            rotation_step(gate, &mut boxes, &mut tape, &too_heavy, &pruned_here)?;
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
    Ok((tape, boxes.words, boxes.alive))
}

/// One rotation gate: pair every word it changes with the word it turns into.
/// A word the pruners reject is dropped instead.
fn rotation_step<const NW: usize>(
    gate: &GateSpec<NW>,
    boxes: &mut Boxes<NW>,
    tape: &mut Tape,
    too_heavy: &impl Fn(&PauliKey<NW>) -> bool,
    pruned: &impl Fn(&PauliKey<NW>) -> bool,
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
        let partner = boxes.get_or_new(set_label(x, z, gate.wire0, out));
        // `word -> cos·word + sign·sin·partner`: with sign +1 the word is
        // `a`; with sign -1 the partner is (the rule is antisymmetric).
        let pair = if sign > 0 { [b as u32, partner] } else { [partner, b as u32] };
        boxes.stamp[pair[0] as usize] = step;
        boxes.stamp[pair[1] as usize] = step;
        tape.pairs.push(pair);
    }

    let last = tape.pairs.len() as u32;
    if last > first {
        tape.steps.push(Step::Rotate { angle, first, last });
    }
    Ok(())
}
