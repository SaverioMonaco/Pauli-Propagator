//! The tape engine: the propagation is compiled once into a list of steps,
//! which then evaluate the expectation value for any parameters.
//!
//! Every Pauli word the propagation meets gets a numbered *box* holding its
//! coefficient. A rotation mixes pairs of boxes, and with each pair written in
//! the right order every pair of every rotation is updated the same way. A
//! pair `[a, b]` is one where the gate turns `a` into `cos·a + sin·b` (RX:
//! `a` has Z and `b` has Y on the gate's qubit; RY: X and Z; RZ: Y and X). The
//! rule for `b` is then always `cos·b − sin·a`, so evaluating a pair is
//!
//! ```text
//! new a = cos·a − sin·b
//! new b = sin·a + cos·b
//! ```
//!
//! with no gate kind or sign stored per pair.
//!
//! Example: RX(θ0) on q0, RX(θ1) on q1, CNOT(q0 -> q1), RX(θ2) on q0,
//! RX(θ3) on q1, observable Z on q1:
//!
//! ```text
//! boxes:   0 = IZ (ZZ after the CNOT)   1 = IY (ZY after the CNOT)   2 = YZ   3 = YY
//! init:    [(0, 1.0)]
//! steps:   [Step(Param(3), 0..1),   RX(θ3) on q1
//!           Step(Param(1), 1..2),   RX(θ1) on q1
//!           Step(Param(0), 2..4)]   RX(θ0) on q0
//! pairs:   [[0, 1], [0, 1], [0, 2], [1, 3]]
//! readout: [0]
//! ```
//!
//! RX(θ2) touches no word and the CNOT only renames boxes while compiling,
//! so neither leaves a step.

// Nothing uses the tape until it can be evaluated (A3).
#![allow(dead_code)]

/// The angle of one rotation step.
pub(crate) enum Angle {
    /// A trainable parameter: its index in the parameter vector.
    Param(u32),
    /// A fixed angle, already turned into its cosine and sine.
    Fixed { cos: f64, sin: f64 },
}

/// One rotation gate: its angle, and the pairs of boxes it mixes,
/// `pairs[first..last]`.
pub(crate) struct Step {
    pub(crate) angle: Angle,
    pub(crate) first: u32,
    pub(crate) last: u32,
}

/// A compiled propagation of one observable.
pub(crate) struct Tape {
    /// How many boxes an evaluation needs.
    pub(crate) n_boxes: u32,
    /// The observable: (box, starting value) for each of its Pauli words.
    pub(crate) init: Vec<(u32, f64)>,
    /// The rotation steps, in the order they are applied (circuit order reversed).
    pub(crate) steps: Vec<Step>,
    /// The pairs `[a, b]` of all steps, one step after another (8 bytes each).
    pub(crate) pairs: Vec<[u32; 2]>,
    /// The boxes added up at the end: the words with only I and Z.
    pub(crate) readout: Vec<u32>,
}
