//! Compiling a tape: the propagation done once, with box numbers in place of
//! coefficients.
//!
//! The gates are walked in reverse (Heisenberg picture), exactly like the
//! sum-of-terms engine, but a Pauli word carries no terms: only its box
//! number. For every rotation, each word the gate changes is paired with the
//! word it turns into, and the pair is written to the tape as `[a, b]` (see
//! the module doc in `mod.rs` for the order).

use super::{Angle, Step, Tape};
use crate::gates::*;
use crate::pauli::*;
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
}

impl<const NW: usize> Boxes<NW> {
    fn new() -> Self {
        Boxes { words: Vec::new(), index: FxHashMap::default(), stamp: Vec::new() }
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
        b
    }
}

/// The rule table of a one-qubit rotation, or `None` for a gate the tape
/// cannot compile yet.
fn rotation_rule(kind: u8) -> Option<fn(u8) -> Option<(u8, i8)>> {
    match kind {
        RX => Some(rx_rule),
        RY => Some(ry_rule),
        RZ => Some(rz_rule),
        _ => None,
    }
}

/// Compile one observable through `gates` (in circuit order). Returns the
/// tape and, for debugging, the word of every box.
pub(crate) fn compile<const NW: usize>(
    gates: &[GateSpec<NW>],
    observable: &[(PauliKey<NW>, f64)],
) -> Result<(Tape, Vec<PauliKey<NW>>), String> {
    let mut boxes = Boxes::<NW>::new();
    let mut tape = Tape {
        n_boxes: 0,
        init: Vec::new(),
        steps: Vec::new(),
        pairs: Vec::new(),
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

    for gate in gates.iter().rev() {
        let rule = rotation_rule(gate.kind)
            .ok_or_else(|| format!("the tape cannot compile gate kind {} yet", gate.kind))?;
        if gate.param < 0 {
            return Err(format!(
                "the tape cannot compile a rotation (gate kind {}) with a fixed angle yet",
                gate.kind
            ));
        }
        let step = tape.steps.len() as u32 + 1;
        let first = tape.pairs.len() as u32;

        // Only the words that exist before this gate can be changed by it; the
        // partners it creates are already in a pair of this step.
        for b in 0..boxes.len() {
            if boxes.stamp[b] == step {
                continue;
            }
            let (x, z) = boxes.words[b];
            let Some((out, sign)) = rule(pauli_label(&x, &z, gate.wire0)) else {
                continue; // the gate commutes with this word
            };
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
            tape.steps.push(Step { angle: Angle::Param(gate.param as u32), first, last });
        }
    }

    tape.n_boxes = boxes.len() as u32;
    tape.readout = (0..boxes.len() as u32)
        .filter(|&b| mask_is_zero(&boxes.words[b as usize].0))
        .collect();
    Ok((tape, boxes.words))
}
