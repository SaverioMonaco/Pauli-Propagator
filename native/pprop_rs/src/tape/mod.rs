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

mod compile;

pub(crate) use compile::*;

use crate::gates::GateSpec;
use crate::pauli::*;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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

/// What `tape_debug` returns to Python: `n_boxes`, `init`, the steps as
/// `(parameter index, first pair, last pair)`, the pairs, the readout, and
/// the word of every box as `(x words, z words)`.
pub(crate) type TapeDebug = (
    u32,
    Vec<(u32, f64)>,
    Vec<(u32, u32, u32)>,
    Vec<(u32, u32)>,
    Vec<u32>,
    Vec<(Vec<u64>, Vec<u64>)>,
);

/// Test-only: compile one observable and return the tape as plain lists, so
/// tests can check the compiler before the tape can be evaluated.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tape_debug_impl<const NW: usize>(
    gate_kind: Vec<u8>,
    gate_wire0: Vec<u32>,
    gate_wire1: Vec<i64>,
    gate_param: Vec<i64>,
    gate_fixed: Vec<Option<f64>>,
    observable: Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>,
) -> PyResult<TapeDebug> {
    let gates: Vec<GateSpec<NW>> = (0..gate_kind.len())
        .map(|i| GateSpec::new(gate_kind[i], gate_wire0[i], gate_wire1[i], gate_param[i], gate_fixed[i]))
        .collect();
    let mut obs = Vec::with_capacity(observable.len());
    for (xw, zw, coeff, sin_idx, cos_idx) in observable {
        if !sin_idx.is_empty() || !cos_idx.is_empty() {
            return Err(PyValueError::new_err(
                "the tape needs an observable with plain coefficients (no sin/cos factors)",
            ));
        }
        obs.push(((words_to_array::<NW>(&xw), words_to_array::<NW>(&zw)), coeff));
    }
    let (tape, words) = compile(&gates, &obs).map_err(PyValueError::new_err)?;
    let steps = tape
        .steps
        .iter()
        .map(|s| match s.angle {
            Angle::Param(p) => (p, s.first, s.last),
            Angle::Fixed { .. } => unreachable!("A2 compiles trainable rotations only"),
        })
        .collect();
    Ok((
        tape.n_boxes,
        tape.init,
        steps,
        tape.pairs.iter().map(|p| (p[0], p[1])).collect(),
        tape.readout,
        words.iter().map(|(x, z)| (x.to_vec(), z.to_vec())).collect(),
    ))
}
