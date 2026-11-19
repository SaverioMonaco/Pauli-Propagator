//! Evaluating a tape: one pass over its steps, on one array of box values.

use super::{Angle, Step, Tape};

impl Tape {
    /// The expectation value at the given `sin(θ)` and `cos(θ)`.
    ///
    /// Every box starts at 0 except the observable's. A rotation step mixes its
    /// pairs; no box appears twice in one step (the compiler's stamps make
    /// sure of that), so a pair only reads and writes its own two boxes and
    /// can be updated in place. A negate step flips the sign of its boxes.
    pub(crate) fn value(&self, sins: &[f64], coss: &[f64]) -> f64 {
        let mut v = vec![0.0f64; self.n_boxes as usize];
        for &(b, c) in &self.init {
            v[b as usize] = c;
        }
        for step in &self.steps {
            match *step {
                Step::Rotate { ref angle, first, last } => {
                    let (c, s) = match *angle {
                        Angle::Param(p) => (coss[p as usize], sins[p as usize]),
                        Angle::Fixed { cos, sin } => (cos, sin),
                    };
                    for &[a, b] in &self.pairs[first as usize..last as usize] {
                        let (a, b) = (a as usize, b as usize);
                        let (va, vb) = (v[a], v[b]);
                        v[a] = c * va - s * vb;
                        v[b] = s * va + c * vb;
                    }
                }
                Step::Negate { first, last } => {
                    for &b in &self.negs[first as usize..last as usize] {
                        v[b as usize] = -v[b as usize];
                    }
                }
            }
        }
        self.readout.iter().map(|&b| v[b as usize]).sum()
    }

    /// How many parameters an evaluation must be given: one more than the
    /// largest index a step uses.
    pub(crate) fn n_params(&self) -> usize {
        self.steps
            .iter()
            .filter_map(|s| match *s {
                Step::Rotate { angle: Angle::Param(p), .. } => Some(p as usize + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}
