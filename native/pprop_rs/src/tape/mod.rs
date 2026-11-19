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
//! steps:   [Rotate(Param(3), 0..1),   RX(θ3) on q1
//!           Rotate(Param(1), 1..2),   RX(θ1) on q1
//!           Rotate(Param(0), 2..4)]   RX(θ0) on q0
//! pairs:   [[0, 1], [0, 1], [0, 2], [1, 3]]
//! readout: [0]
//! ```
//!
//! RX(θ2) touches no word and the CNOT only renames boxes while compiling,
//! so neither leaves a step.
//!
//! A Clifford gate (H, S, SX, CNOT, CY, CZ, SWAP) never mixes words: it turns
//! each word into exactly one other word, sometimes with a minus sign. The
//! compiler renames the boxes (a box keeps its number, its word changes), and
//! the minus signs become a `Negate` step listing the boxes to multiply by −1.
//!
//! A controlled rotation mixes groups of up to 4 words, whose members can
//! have different Pauli weights. Such a gate is a `Block` step: each group is
//! a `Block` of up to 4 boxes, mixed by the small matrix of its `BlockProg`.
//! A member that does not exist (too heavy for k1, say) is `NONE`: it is read
//! as 0 and not written.

mod compile;
mod eval;

pub(crate) use compile::*;

use crate::gates::GateSpec;
use crate::pauli::*;
use crate::pybuf::read_f64;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// The angle of one rotation step.
pub(crate) enum Angle {
    /// A trainable parameter: its index in the parameter vector.
    Param(u32),
    /// A fixed angle, already turned into its cosine and sine.
    Fixed { cos: f64, sin: f64 },
}

/// One step of the tape.
pub(crate) enum Step {
    /// A rotation: its angle, and the pairs of boxes it mixes, `pairs[first..last]`.
    Rotate { angle: Angle, first: u32, last: u32 },
    /// The minus signs of a Clifford gate: negate the boxes `negs[first..last]`.
    Negate { first: u32, last: u32 },
    /// Groups of up to 4 boxes, `blocks[first..last]`, each mixed by its
    /// `BlockProg` at this step's angle (a controlled rotation).
    Block { angle: Angle, first: u32, last: u32 },
}

/// A member of a block with no box: read as 0, not written.
pub(crate) const NONE: u32 = u32::MAX;

/// How the members of one kind of block mix: each entry
/// `(out member, in member, coeff, n_sin, n_cos)` adds
/// `coeff · sin^n_sin · cos^n_cos · old[in]` to `new[out]`.
pub(crate) struct BlockProg {
    pub(crate) size: u8,
    pub(crate) entries: Vec<(u8, u8, f64, u8, u8)>,
}

/// A group of up to `size` boxes mixed together, `prog` its `BlockProg`.
pub(crate) struct Block {
    pub(crate) prog: u32,
    pub(crate) members: [u32; 4],
}

/// A compiled propagation of one observable.
pub(crate) struct Tape {
    /// How many boxes an evaluation needs.
    pub(crate) n_boxes: u32,
    /// The observable: (box, starting value) for each of its Pauli words.
    pub(crate) init: Vec<(u32, f64)>,
    /// The steps, in the order they are applied (circuit order reversed).
    pub(crate) steps: Vec<Step>,
    /// The pairs `[a, b]` of all rotation steps, one step after another (8 bytes each).
    pub(crate) pairs: Vec<[u32; 2]>,
    /// The boxes of all negate steps, one step after another.
    pub(crate) negs: Vec<u32>,
    /// The blocks of all block steps, one step after another (20 bytes each).
    pub(crate) blocks: Vec<Block>,
    /// The kinds of block the blocks refer to (a handful per gate kind).
    pub(crate) block_progs: Vec<BlockProg>,
    /// The boxes added up at the end: the words with only I and Z.
    pub(crate) readout: Vec<u32>,
}

/// What `tape_debug` returns: the tape and, per box, its word and whether it
/// is still alive at the end (false once dropped). Handed to Python as a
/// dict, so new fields never break existing callers.
pub(crate) struct TapeDebug {
    tape: Tape,
    words: Vec<(Vec<u64>, Vec<u64>)>,
    alive: Vec<bool>,
}

impl TapeDebug {
    /// The dict: `n_boxes`, `init`, `steps` (`("rotate", parameter, first,
    /// last)`, `("fixed", -1, ...)`, `("negate", -1, ...)`, `("block",
    /// parameter, ...)` or `("block_fixed", -1, ...)`), `pairs`, `negs`,
    /// `blocks` (`(prog, members)` with -1 for `NONE`), `block_progs`
    /// (`(size, entries)`), `readout`, `words` (`(x words, z words)`) and
    /// `alive`.
    pub(crate) fn into_dict(self, py: Python<'_>) -> PyResult<PyObject> {
        use pyo3::types::PyDict;
        let t = self.tape;
        let steps: Vec<(&str, i64, u32, u32)> = t
            .steps
            .iter()
            .map(|s| match *s {
                Step::Rotate { angle: Angle::Param(p), first, last } => ("rotate", p as i64, first, last),
                Step::Rotate { angle: Angle::Fixed { .. }, first, last } => ("fixed", -1, first, last),
                Step::Negate { first, last } => ("negate", -1, first, last),
                Step::Block { angle: Angle::Param(p), first, last } => ("block", p as i64, first, last),
                Step::Block { angle: Angle::Fixed { .. }, first, last } => ("block_fixed", -1, first, last),
            })
            .collect();
        let blocks: Vec<(u32, Vec<i64>)> = t
            .blocks
            .iter()
            .map(|b| {
                let size = t.block_progs[b.prog as usize].size as usize;
                let members = b.members[..size].iter().map(|&m| if m == NONE { -1 } else { m as i64 }).collect();
                (b.prog, members)
            })
            .collect();
        let progs: Vec<(u8, Vec<(u8, u8, f64, u8, u8)>)> =
            t.block_progs.iter().map(|p| (p.size, p.entries.clone())).collect();
        let d = PyDict::new_bound(py);
        d.set_item("n_boxes", t.n_boxes)?;
        d.set_item("init", t.init)?;
        d.set_item("steps", steps)?;
        d.set_item("pairs", t.pairs.iter().map(|p| (p[0], p[1])).collect::<Vec<_>>())?;
        d.set_item("negs", t.negs)?;
        d.set_item("blocks", blocks)?;
        d.set_item("block_progs", progs)?;
        d.set_item("readout", t.readout)?;
        d.set_item("words", self.words)?;
        d.set_item("alive", self.alive)?;
        Ok(d.into_any().unbind())
    }
}

/// Test-only: compile one observable and return the tape (see `TapeDebug`),
/// so tests can look at what the compiler wrote.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tape_debug_impl<const NW: usize>(
    gate_kind: Vec<u8>,
    gate_wire0: Vec<u32>,
    gate_wire1: Vec<i64>,
    gate_param: Vec<i64>,
    gate_fixed: Vec<Option<f64>>,
    observable: Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>,
    k1: i64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
) -> PyResult<TapeDebug> {
    let gates = gates_from_arrays::<NW>(&gate_kind, &gate_wire0, &gate_wire1, &gate_param, &gate_fixed);
    let obs = observable_from_spec::<NW>(observable)?;
    let (tape, words, alive) = compile(&gates, &obs, k1, use_dead_qubit_pruner, use_xy_weight_pruner)
        .map_err(PyValueError::new_err)?;
    let words = words.iter().map(|(x, z)| (x.to_vec(), z.to_vec())).collect();
    Ok(TapeDebug { tape, words, alive })
}

/// The gates from the arrays Python passes, in circuit order.
fn gates_from_arrays<const NW: usize>(
    kind: &[u8],
    wire0: &[u32],
    wire1: &[i64],
    param: &[i64],
    fixed: &[Option<f64>],
) -> Vec<GateSpec<NW>> {
    (0..kind.len())
        .map(|i| GateSpec::new(kind[i], wire0[i], wire1[i], param[i], fixed[i]))
        .collect()
}

/// One observable as `(word, coefficient)` pairs, from the format Python
/// passes (`(x words, z words, coeff, sin indices, cos indices)` per term).
/// The tape needs plain coefficients.
fn observable_from_spec<const NW: usize>(
    spec: Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>,
) -> PyResult<Vec<(PauliKey<NW>, f64)>> {
    let mut obs = Vec::with_capacity(spec.len());
    for (xw, zw, coeff, sin_idx, cos_idx) in spec {
        if !sin_idx.is_empty() || !cos_idx.is_empty() {
            return Err(PyValueError::new_err(
                "the tape needs an observable with plain coefficients (no sin/cos factors)",
            ));
        }
        obs.push(((words_to_array::<NW>(&xw), words_to_array::<NW>(&zw)), coeff));
    }
    Ok(obs)
}

/// A compiled tape, as Python sees it: `program.eval(sins, coss)` gives the
/// expectation value at `sins = sin(θ)`, `coss = cos(θ)`.
#[pyclass]
pub(crate) struct TapeProgram {
    tape: Tape,
    n_params: usize,
}

#[pymethods]
impl TapeProgram {
    /// Expectation value at the given `sin(θ)` and `cos(θ)` (float64 arrays,
    /// at least as long as the largest parameter index the tape uses + 1).
    fn eval(&self, py: Python<'_>, sins: &Bound<'_, PyAny>, coss: &Bound<'_, PyAny>) -> PyResult<f64> {
        let sins = read_f64(sins)?;
        let coss = read_f64(coss)?;
        if sins.len() < self.n_params || coss.len() < self.n_params {
            return Err(PyValueError::new_err(format!(
                "sins and coss need at least {} entries", self.n_params)));
        }
        Ok(py.allow_threads(|| self.tape.value(&sins, &coss)))
    }

    /// Number of boxes (Pauli words) an evaluation works on.
    #[getter]
    fn n_boxes(&self) -> u32 {
        self.tape.n_boxes
    }

    /// Number of steps (rotations and negations).
    #[getter]
    fn n_steps(&self) -> usize {
        self.tape.steps.len()
    }

    /// Number of pairs, over all steps (one 2x2 update each per evaluation).
    #[getter]
    fn n_pairs(&self) -> usize {
        self.tape.pairs.len()
    }
}

/// Compile every observable into a `TapeProgram`.
pub(crate) fn compile_tapes_impl<const NW: usize>(
    gate_kind: Vec<u8>,
    gate_wire0: Vec<u32>,
    gate_wire1: Vec<i64>,
    gate_param: Vec<i64>,
    gate_fixed: Vec<Option<f64>>,
    paulidicts: Vec<Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>>,
    k1: i64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
) -> PyResult<Vec<TapeProgram>> {
    let gates = gates_from_arrays::<NW>(&gate_kind, &gate_wire0, &gate_wire1, &gate_param, &gate_fixed);
    let mut programs = Vec::with_capacity(paulidicts.len());
    for spec in paulidicts {
        let obs = observable_from_spec::<NW>(spec)?;
        let (tape, _words, _alive) = compile(&gates, &obs, k1, use_dead_qubit_pruner, use_xy_weight_pruner)
            .map_err(PyValueError::new_err)?;
        let n_params = tape.n_params();
        programs.push(TapeProgram { tape, n_params });
    }
    Ok(programs)
}
