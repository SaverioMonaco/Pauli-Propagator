//! Rust implementation of pprop's Heisenberg propagation core.
//!
//! It covers every gate in `pprop.gates` (RX/RY/RZ, H/S/SX/T, SWAP, CNOT/CY/CZ,
//! CRX/CRY/CRZ), both exact pruners (`DeadQubitPruner`, `XYWeightPruner`), and
//! all three truncations (`WeightTruncation`, `FrequencyTruncation`,
//! `CoefficientTruncation`). The evolution rule tables below are transcribed
//! 1:1 from the `rule` dicts in `pprop/gates/*.py`, which remain in the Python
//! source as the human-readable reference these tables are checked against,
//! entry-for-entry, by `tests/test_rule_tables.py` (via
//! `evolve_single_gate_debug` below). `tests/test_backends.py` covers
//! correctness at the full-circuit level instead (random circuits vs. PennyLane
//! directly).
//!
//! How a Pauli word is stored as bits is described in `pauli.rs`.

// Shards move term lists between threads. Use mimalloc for the extension's
// cross-thread allocations; allocator choice is independent of threading.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

mod gates;
mod pauli;
mod pruning;
mod terms;
mod threads;
use pauli::*;
use terms::*;

/// Dispatches to the `$f::<NW>` monomorphization matching `$nw` (a
/// `words_needed(num_qubits)` result), or returns a `PyValueError` if `$nw`
/// exceeds every supported size. Add a literal here (and nowhere else) to
/// raise the qubit ceiling.
///
/// `$f` is a bare function path (not a call), and the numeric literal is
/// substituted directly into each arm's turbofish rather than bound to a
/// shared identifier - macro hygiene would make a `const NW` defined here
/// invisible to a `NW` token written at the call site, so each arm must
/// spell its word count out explicitly.
macro_rules! dispatch_nw {
    ($nw:expr, $num_qubits:expr, $f:ident, ( $($args:expr),* $(,)? )) => {
        match $nw {
            1 => $f::<1>($($args),*),
            2 => $f::<2>($($args),*),
            4 => $f::<4>($($args),*),
            8 => $f::<8>($($args),*),
            16 => $f::<16>($($args),*),
            32 => $f::<32>($($args),*),
            64 => $f::<64>($($args),*),
            128 => $f::<128>($($args),*),
            other => return Err(PyValueError::new_err(format!(
                "num_qubits={} needs {} u64 words per Pauli-word plane, which exceeds \
                 the maximum supported 128 words (8192 qubits) in this build of pprop_rs. \
                 Add a larger literal to dispatch_nw! in lib.rs if you need more.",
                $num_qubits, other
            ))),
        }
    };
}

#[pyfunction]
#[pyo3(signature = (
    num_qubits,
    gate_kind, gate_wire0, gate_wire1, gate_param, gate_fixed,
    k1, k2, coeff_threshold,
    use_dead_qubit_pruner, use_xy_weight_pruner,
    paulidicts,
    n_threads = 1,
    sine_order = false,
))]
#[allow(clippy::too_many_arguments)]
fn propagate_batch(
    py: Python<'_>,
    num_qubits: u32,
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
) -> PyResult<Vec<Vec<(f64, Vec<u32>, Vec<u32>)>>> {
    let nw = words_needed(num_qubits);
    py.allow_threads(move || {
        dispatch_nw!(
            nw, num_qubits, propagate_batch_impl,
            (
                gate_kind, gate_wire0, gate_wire1, gate_param, gate_fixed,
                k1, k2, coeff_threshold, use_dead_qubit_pruner, use_xy_weight_pruner,
                paulidicts, n_threads, sine_order,
            )
        )
    })
}

/// Test-only helper: evolve a single Pauli word through one gate and return
/// the *unfiltered* resulting map (no zerobracket filtering, unlike
/// `propagate_batch`). Used by `tests/test_rule_tables.py` to cross-check
/// every gate's `rule` dict in `pprop/gates/*.py` directly against this
/// file's rule tables, term-for-term, independent of whether the output
/// happens to land in the Z/I subspace. `fixed` is the constant angle of a
/// non-trainable rotation (see `GateSpec::fixed`); leave it `None` for a
/// trainable one.
#[pyfunction]
#[pyo3(signature = (num_qubits, kind, wire0, wire1, param, x, z, terms, fixed = None))]
#[allow(clippy::too_many_arguments)]
fn evolve_single_gate_debug(
    num_qubits: u32,
    kind: u8,
    wire0: u32,
    wire1: i64,
    param: i64,
    x: Vec<u64>,
    z: Vec<u64>,
    terms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    fixed: Option<f64>,
) -> PyResult<Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>> {
    let nw = words_needed(num_qubits);
    Ok(dispatch_nw!(
        nw, num_qubits, evolve_single_gate_debug_impl,
        (kind, wire0, wire1, param, fixed, &x, &z, terms)
    ))
}

#[pymodule]
fn pprop_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(propagate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_layout, m)?)?;
    m.add_class::<Evaluator>()?;
    m.add_function(wrap_pyfunction!(evolve_single_gate_debug, m)?)?;
    Ok(())
}
