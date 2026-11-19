//! Evaluating the sum-of-terms expression: the flat ("ragged") layout of the
//! terms, and the compiled `Evaluator` that computes value and gradient.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use crate::pybuf::*;

/// The CSR layout `pprop.propagator.evaluator.build_ragged_arrays` builds
/// (coefficients, lookup-table index of every factor, factors per term), as
/// native-endian float64/int64 buffers for `numpy.frombuffer`. Same layout,
/// entry for entry; this just skips the per-factor Python loop.
#[pyfunction]
pub(crate) fn ragged_layout<'py>(
    py: Python<'py>,
    expr: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_params: u32,
) -> (
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
) {
    let sentinel = num_params as i64;
    let cos_offset = num_params as i64 + 1;
    let mut coeffs: Vec<u8> = Vec::with_capacity(8 * expr.len());
    let mut cnt: Vec<u8> = Vec::with_capacity(8 * expr.len());
    let mut idx: Vec<u8> = Vec::new();
    for (coeff, sin_idx, cos_idx) in &expr {
        coeffs.extend_from_slice(&coeff.to_ne_bytes());
        for &j in sin_idx {
            idx.extend_from_slice(&(j as i64).to_ne_bytes());
        }
        for &j in cos_idx {
            idx.extend_from_slice(&(cos_offset + j as i64).to_ne_bytes());
        }
        let mut n = (sin_idx.len() + cos_idx.len()) as i64;
        if n == 0 {
            idx.extend_from_slice(&sentinel.to_ne_bytes());
            n = 1;
        }
        cnt.extend_from_slice(&n.to_ne_bytes());
    }
    (
        PyByteArray::new_bound(py, &coeffs),
        PyByteArray::new_bound(py, &idx),
        PyByteArray::new_bound(py, &cnt),
    )
}

/// Add `term` to the accumulator slot of every factor in `run`, spreading
/// consecutive factors over `C` interleaved copies of the accumulator so that
/// a recurring parameter does not stall on the previous read-modify-write.
///
/// The writes skip bounds checking: `Evaluator::new` rejects any index that
/// does not address one copy, and `acc` is `C * width` long.
#[inline]
fn scatter<const C: usize>(run: &[u32], term: f64, acc: &mut [f64], width: usize) {
    let mut chunks = run.chunks_exact(C);
    for chunk in &mut chunks {
        for (copy, &entry) in chunk.iter().enumerate() {
            unsafe { *acc.get_unchecked_mut(copy * width + entry as usize) += term };
        }
    }
    for &entry in chunks.remainder() {
        unsafe { *acc.get_unchecked_mut(entry as usize) += term };
    }
}

/// Compiled evaluator for one propagated expression, mirroring the NumPy
/// implementation in `pprop.propagator.evaluator` (see `build_ragged_arrays`
/// there for how `coeffs`/`idx`/`cnt` are produced and what the index space
/// means). Everything below is that same arithmetic with the intermediate
/// arrays removed: the terms are walked once, factor by factor, keeping the
/// running products in registers rather than materialising one temporary per
/// pass.
///
/// Index space, with `P = num_params`: a factor `sin(theta_k)` is `k`, a
/// factor `cos(theta_k)` is `P + 1 + k`, and `P` itself is a sentinel whose
/// value is 1 (it stands in for a term with no factors left). Powers are
/// repeated indices, so no exponent is ever evaluated.
#[pyclass]
pub struct Evaluator {
    coeffs: Vec<f64>,
    idx: Vec<u32>,
    /// Start of each term's run in `idx`, with a closing entry: length is
    /// `coeffs.len() + 1`.
    off: Vec<usize>,
    num_params: usize,
    /// Longest single term, i.e. the scratch space the exact path needs.
    max_run: usize,
    /// How many interleaved copies of the gradient accumulator to keep -
    /// 4, 2 or 1, whichever fits in L1 for this parameter count. See
    /// `scatter`.
    acc_copies: usize,
    tol: f64,
}

impl Evaluator {
    /// `[sin(theta_0..P-1), 1.0, cos(theta_0..P-1)]`, the table `idx` reads.
    fn table(&self, sins: &[f64], coss: &[f64]) -> Vec<f64> {
        let p = self.num_params;
        let mut table = Vec::with_capacity(2 * p + 1);
        table.extend_from_slice(sins);
        table.push(1.0);
        table.extend_from_slice(coss);
        table
    }

    /// Product of one term's factors. Four independent accumulators, so the
    /// multiplies pipeline instead of forming one dependency chain per term.
    ///
    /// The table lookups skip bounds checking: `new` rejects any index that
    /// does not address `table`, which is the only place `idx` is set.
    #[inline]
    fn term_product(&self, table: &[f64], start: usize, end: usize) -> f64 {
        debug_assert!(table.len() == 2 * self.num_params + 1);
        let run = &self.idx[start..end];
        let (mut a, mut b, mut c, mut d) = (1.0, 1.0, 1.0, 1.0);
        let mut chunks = run.chunks_exact(4);
        for chunk in &mut chunks {
            unsafe {
                a *= *table.get_unchecked(chunk[0] as usize);
                b *= *table.get_unchecked(chunk[1] as usize);
                c *= *table.get_unchecked(chunk[2] as usize);
                d *= *table.get_unchecked(chunk[3] as usize);
            }
        }
        for &entry in chunks.remainder() {
            a *= unsafe { *table.get_unchecked(entry as usize) };
        }
        (a * b) * (c * d)
    }

    /// Gradient via the logarithmic derivative: every factor on parameter `k`
    /// contributes `term * cot(theta_k)` (or `-term * tan(theta_k)` for a
    /// cosine), which depends only on `k`. So the factors just accumulate
    /// term values per parameter and the cot/tan multiply happens once over
    /// the parameter vector. Requires every angle to sit away from a zero of
    /// sin/cos; `grad_exact` handles the rest.
    fn grad_fast(&self, sins: &[f64], coss: &[f64], grad: &mut [f64]) -> f64 {
        let p = self.num_params;
        let table = self.table(sins, coss);
        let width = 2 * p + 1;
        // Consecutive factors are scattered into *different* copies of the
        // accumulator, so a parameter that recurs within a few factors does
        // not stall waiting on the previous read-modify-write. The copies are
        // summed back together at the end.
        let copies = self.acc_copies;
        let mut acc = vec![0.0f64; copies * width];
        let mut value = 0.0;

        for t in 0..self.coeffs.len() {
            let (start, end) = (self.off[t], self.off[t + 1]);
            let term = self.coeffs[t] * self.term_product(&table, start, end);
            value += term;
            let run = &self.idx[start..end];
            match copies {
                4 => scatter::<4>(run, term, &mut acc, width),
                2 => scatter::<2>(run, term, &mut acc, width),
                _ => scatter::<1>(run, term, &mut acc, width),
            }
        }

        for k in 0..p {
            let mut sin_total = 0.0;
            let mut cos_total = 0.0;
            for c in 0..copies {
                sin_total += acc[c * width + k];
                cos_total += acc[c * width + p + 1 + k];
            }
            grad[k] = (coss[k] / sins[k]) * sin_total - (sins[k] / coss[k]) * cos_total;
        }
        value
    }

    /// Gradient without dividing by anything: for each term, walk its factors
    /// forwards recording partial products, then backwards multiplying by the
    /// running suffix, which gives the product of all *other* factors exactly.
    /// Costs a second pass over each term and some scratch, and is used only
    /// when an angle sits at (or extremely near) a zero of sin or cos, where
    /// the cot/tan form above is singular. Repeated indices come out right on
    /// their own: p copies each contribute the product of the other p-1.
    fn grad_exact(&self, sins: &[f64], coss: &[f64], grad: &mut [f64]) -> f64 {
        let p = self.num_params;
        let table = self.table(sins, coss);
        let mut prefix = vec![0.0f64; self.max_run];
        let mut value = 0.0;

        for t in 0..self.coeffs.len() {
            let (start, end) = (self.off[t], self.off[t + 1]);
            let coeff = self.coeffs[t];

            let mut running = 1.0;
            for (j, i) in (start..end).enumerate() {
                prefix[j] = running;
                running *= table[self.idx[i] as usize];
            }
            value += coeff * running;

            let mut suffix = 1.0;
            for (j, i) in (start..end).enumerate().rev() {
                let entry = self.idx[i] as usize;
                let excluding = coeff * prefix[j] * suffix;
                if entry < p {
                    grad[entry] += excluding * coss[entry];        // d sin / d theta
                } else if entry > p {
                    let k = entry - (p + 1);
                    grad[k] -= excluding * sins[k];                // d cos / d theta
                }
                suffix *= table[entry];
            }
        }
        value
    }
}

#[pymethods]
impl Evaluator {
    /// `coeffs`, `idx` and `cnt` are the arrays `build_ragged_arrays` returns,
    /// as buffers (`float64`, `uint32`, `uint32`).
    #[new]
    #[pyo3(signature = (coeffs, idx, cnt, num_params, tol = 1e-6))]
    fn new(
        coeffs: &Bound<'_, PyAny>,
        idx: &Bound<'_, PyAny>,
        cnt: &Bound<'_, PyAny>,
        num_params: usize,
        tol: f64,
    ) -> PyResult<Self> {
        let coeffs = read_f64(coeffs)?;
        let idx = read_u32(idx)?;
        let cnt = read_u32(cnt)?;
        if cnt.len() != coeffs.len() {
            return Err(PyValueError::new_err("coeffs and cnt must have equal length"));
        }

        let mut off = Vec::with_capacity(cnt.len() + 1);
        let mut total = 0usize;
        off.push(0);
        for &c in &cnt {
            total += c as usize;
            off.push(total);
        }
        if total != idx.len() {
            return Err(PyValueError::new_err("cnt does not add up to len(idx)"));
        }
        let limit = 2 * num_params + 1;
        if idx.iter().any(|&i| i as usize >= limit) {
            return Err(PyValueError::new_err("factor index out of range"));
        }

        let max_run = cnt.iter().map(|&c| c as usize).max().unwrap_or(0);
        // Keep the interleaved accumulators inside ~16 KB, i.e. half of a
        // typical L1, or the trick costs more in misses than it saves. Wide
        // parameter vectors settle for two copies, or one.
        let width = 2 * num_params + 1;
        let bytes = width * std::mem::size_of::<f64>();
        let acc_copies = [4, 2, 1].into_iter().find(|c| c * bytes <= 16 * 1024).unwrap_or(1);
        Ok(Evaluator { coeffs, idx, off, num_params, max_run, acc_copies, tol })
    }

    /// Expectation value at the given `sin(theta)`/`cos(theta)`.
    fn eval(&self, py: Python<'_>, sins: &Bound<'_, PyAny>, coss: &Bound<'_, PyAny>)
        -> PyResult<f64>
    {
        let sins = self.read_angles(sins)?;
        let coss = self.read_angles(coss)?;
        Ok(py.allow_threads(|| {
            let table = self.table(&sins, &coss);
            (0..self.coeffs.len())
                .map(|t| self.coeffs[t] * self.term_product(&table, self.off[t], self.off[t + 1]))
                .sum()
        }))
    }

    /// Expectation value, writing the gradient into `out` (`float64`, length
    /// `num_params`) rather than allocating a fresh array per call.
    fn eval_and_grad(
        &self,
        py: Python<'_>,
        sins: &Bound<'_, PyAny>,
        coss: &Bound<'_, PyAny>,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let sins = self.read_angles(sins)?;
        let coss = self.read_angles(coss)?;
        let buffer = PyBuffer::<f64>::get_bound(out)?;
        if buffer.item_count() != self.num_params || buffer.readonly() {
            return Err(PyValueError::new_err("out must be a writable float64 array of length num_params"));
        }

        let mut grad = vec![0.0f64; self.num_params];
        let value = py.allow_threads(|| {
            let singular = sins.iter().chain(coss.iter()).any(|v| v.abs() < self.tol);
            if singular {
                self.grad_exact(&sins, &coss, &mut grad)
            } else {
                self.grad_fast(&sins, &coss, &mut grad)
            }
        });
        buffer.copy_from_slice(py, &grad)?;
        Ok(value)
    }
}

impl Evaluator {
    fn read_angles(&self, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let values = read_f64(obj)?;
        if values.len() != self.num_params {
            return Err(PyValueError::new_err("sins/coss must have length num_params"));
        }
        Ok(values)
    }
}
