"""
JAX-based batched evaluator (``backend="vmap"``).

Fuses *all* observables of a propagated :class:`~pprop.propagator.Propagator`
into a single ``jax.jit`` + ``jax.vmap`` call, built on the same narrow/gathered
array representation as :func:`~pprop.propagator.utils.build_sparse_arrays`
(each observable's arrays are padded to a common shape and stacked along a new
leading "observable" axis).

This removes the Python loop over observables entirely - one compiled call
replaces the whole ``[f(params) for f in self._eval_and_grad_list]``
comprehension used by ``backend="standard"``/``"sparse"``.

Measured trade-off, CPU (see ``tests/test_eval_and_grad_jit_bench.py`` section
5b and ``notebooks/test/sparse_arrays_explained.ipynb`` for the full
writeup): the forward pass here is fast (competitive with or faster than
NumPy), but the gradient's final scatter-add - routing each term's narrow
contribution back into the shared ``(num_params,)`` gradient - is
disproportionately expensive on JAX's CPU backend, making this backend
*slower* than ``backend="sparse"`` on CPU in every case measured so far.

Measured trade-off, GPU (see ``notebooks/test/gpu_backend.ipynb``): the opposite -
that scatter-add is NOT a bottleneck on GPU, and this backend measured
~70-80x faster on GPU than the same batch on CPU, and ~4x faster than
``backend="sparse"`` on CPU, at the 1000-observable scale tested there. GPU
throughput only pays off for a large batch of observables evaluated
together, though - kernel-launch and host/device transfer overhead dominate
for the common case of propagating a handful of observables at a time (the
MMD training workload in ``scripts/generation/train.py``, which propagates
hundreds of observables per round, is the exception, not the rule). This is
why ``backend="sparse"`` stays the default even with a GPU available - see
the ``device`` parameter below to opt into GPU explicitly when your workload
actually batches many observables.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..pauli.sentence import CoeffTerms
from .utils import build_sparse_arrays

# Required for correctness: JAX defaults to float32, which is too coarse once
# gradients from many terms are summed (relevant for Adam-style optimisation).
# Only takes effect if/when this module is actually imported (i.e. backend="vmap"
# is used), so it has no effect on users who never touch this backend.
jax.config.update("jax_enable_x64", True)


def _single_eval(
    thetas: jnp.ndarray,
    coeffs: jnp.ndarray,
    idx_sin: jnp.ndarray,
    pow_sin: jnp.ndarray,
    idx_cos: jnp.ndarray,
    pow_cos: jnp.ndarray,
) -> jnp.ndarray:
    """One observable's expectation value, sparse/gathered form. Mapped over
    the leading observable axis by jax.vmap in make_batched_evaluator."""
    sin_g = jnp.sin(thetas)[idx_sin]
    cos_g = jnp.cos(thetas)[idx_cos]
    sin_pow = jnp.where(pow_sin > 0, sin_g ** pow_sin, 1.0)
    cos_pow = jnp.where(pow_cos > 0, cos_g ** pow_cos, 1.0)
    sin_prod = jnp.prod(sin_pow, axis=-1)
    cos_prod = jnp.prod(cos_pow, axis=-1)
    return jnp.sum(coeffs * sin_prod * cos_prod)


def _resolve_device(device: Optional[str]):
    """Turn a "cpu"/"gpu"/"tpu"/None device request into either a real
    jax.Device or None (None means "leave it to JAX's own default", which is
    GPU/TPU automatically if a matching jaxlib plugin is installed, else
    CPU)."""
    if device is None:
        return None
    try:
        candidates = jax.devices(device)
    except RuntimeError as exc:
        raise ValueError(
            f"device={device!r} requested but JAX reports no such backend "
            f"available in this environment ({exc}). Available backends: "
            f"{sorted({d.platform for d in jax.devices()})}. For GPU, this "
            f"means a CUDA-enabled jaxlib/plugin isn't installed - see "
            f"notebooks/test/gpu_backend.ipynb for the install command used there."
        ) from exc
    return candidates[0]


def make_batched_evaluator(
    exprs: List[CoeffTerms],
    num_params: int,
    device: Optional[str] = None,
) -> Tuple[Callable[[np.ndarray], np.ndarray],
           Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
           Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]]:
    """
    Compile *all* observables' expressions into a single batched JAX callable.

    Unlike :func:`~pprop.propagator.utils.make_evaluator` /
    :func:`~pprop.propagator.utils.make_sparse_evaluator` (one callable per
    observable), this builds ONE callable for the whole list - pad every
    observable's sparse arrays to a common shape, stack them, and
    ``jax.vmap`` a single-observable function across the stacked batch.

    Parameters
    ----------
    exprs : list[CoeffTerms]
        One symbolic expression per observable (``Propagator.exprs`` after
        ``propagate()``).
    num_params : int
        Total number of circuit parameters.
    device : {"cpu", "gpu", "tpu"}, optional
        Which JAX backend to place the batched arrays and compiled functions
        on. ``None`` (default) leaves it to JAX's own default device
        selection (GPU automatically, if a CUDA-enabled jaxlib/plugin is
        installed - see ``notebooks/test/gpu_backend.ipynb``). Passing an explicit
        value is mainly useful to (a) force GPU on a machine where one is
        installed but you don't want it picked implicitly, or (b) force CPU
        even when a GPU is present (e.g. to avoid contending for a shared
        GPU for a workload too small to benefit - see the module docstring).
        Raises :exc:`ValueError` if the requested backend isn't available.

    Returns
    -------
    eval_fn : Callable[[ndarray], ndarray]
        ``eval_fn(theta)`` returns an ``(num_obs,)`` array of expectation
        values (plain NumPy, not a JAX array).
    eval_grad_fn : Callable[[ndarray], Tuple[ndarray, ndarray]]
        ``eval_grad_fn(theta)`` returns ``(vals, grads)`` with shapes
        ``(num_obs,)`` and ``(num_obs, num_params)``.
    raw_eval_and_grad_fn : Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]
        Same computation as ``eval_grad_fn``, but takes and returns raw JAX
        arrays - no ``jnp.asarray``/``np.asarray`` conversion at either end.
        Not useful directly (a plain NumPy ``theta`` in still triggers one
        host->device transfer on the way in), but composable into a larger
        ``jax.jit``'d program with no NumPy round-trip at all - which is
        exactly what :func:`pprop.optimization.adam_gpu` does with it, since
        :meth:`Propagator.eval_and_grad`'s NumPy round-trip on every step is
        what makes plain :func:`~pprop.optimization.adam` disproportionately
        slow for ``backend="vmap"`` (see ``notebooks/test/gpu_backend.ipynb``).
    """
    per_obs = [build_sparse_arrays(e, num_params) for e in exprs]
    n_obs = len(per_obs)

    max_terms = max((coeffs.shape[0] for coeffs, *_ in per_obs), default=1) or 1
    max_ws = max((idx_sin.shape[1] for _, idx_sin, *_ in per_obs), default=1) or 1
    max_wc = max((idx_cos.shape[1] for *_, idx_cos, _ in per_obs), default=1) or 1

    coeffs_b = np.zeros((n_obs, max_terms))
    idx_sin_b = np.zeros((n_obs, max_terms, max_ws), dtype=np.int64)
    pow_sin_b = np.zeros((n_obs, max_terms, max_ws))
    idx_cos_b = np.zeros((n_obs, max_terms, max_wc), dtype=np.int64)
    pow_cos_b = np.zeros((n_obs, max_terms, max_wc))

    for i, (coeffs, idx_sin, pow_sin, idx_cos, pow_cos) in enumerate(per_obs):
        n = coeffs.shape[0]
        coeffs_b[i, :n] = coeffs
        idx_sin_b[i, :n, : idx_sin.shape[1]] = idx_sin
        pow_sin_b[i, :n, : pow_sin.shape[1]] = pow_sin
        idx_cos_b[i, :n, : idx_cos.shape[1]] = idx_cos
        pow_cos_b[i, :n, : pow_cos.shape[1]] = pow_cos

    jax_device = _resolve_device(device)
    # Placing the constant arrays AND compiling under the same device context
    # pins the whole computation there - jax.jit'd functions run on whatever
    # device their (constant) closed-over arrays already live on.
    with jax.default_device(jax_device) if jax_device is not None else nullcontext():
        coeffs_j = jnp.asarray(coeffs_b)
        idx_sin_j = jnp.asarray(idx_sin_b)
        pow_sin_j = jnp.asarray(pow_sin_b)
        idx_cos_j = jnp.asarray(idx_cos_b)
        pow_cos_j = jnp.asarray(pow_cos_b)

        batched_val = jax.jit(jax.vmap(_single_eval, in_axes=(None, 0, 0, 0, 0, 0)))
        batched_val_grad = jax.jit(
            jax.vmap(jax.value_and_grad(_single_eval), in_axes=(None, 0, 0, 0, 0, 0))
        )

    def eval_fn(params: np.ndarray) -> np.ndarray:
        vals = batched_val(jnp.asarray(params), coeffs_j, idx_sin_j, pow_sin_j, idx_cos_j, pow_cos_j)
        return np.asarray(vals)

    def eval_grad_fn(params: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vals, grads = batched_val_grad(
            jnp.asarray(params), coeffs_j, idx_sin_j, pow_sin_j, idx_cos_j, pow_cos_j
        )
        return np.asarray(vals), np.asarray(grads)

    def raw_eval_and_grad_fn(thetas: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        return batched_val_grad(thetas, coeffs_j, idx_sin_j, pow_sin_j, idx_cos_j, pow_cos_j)

    return eval_fn, eval_grad_fn, raw_eval_and_grad_fn
