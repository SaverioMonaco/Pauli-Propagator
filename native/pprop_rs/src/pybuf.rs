//! Reading NumPy arrays (Python buffers) passed in from Python.

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;

pub(crate) fn read_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    PyBuffer::<f64>::get_bound(obj)?.to_vec(obj.py())
}

pub(crate) fn read_u32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    PyBuffer::<u32>::get_bound(obj)?.to_vec(obj.py())
}
