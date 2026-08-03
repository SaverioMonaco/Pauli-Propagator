"""
Utility functions for the Propagator class.

Provides:

- :func:`requires_propagation` -- decorator guarding methods until propagation is done.
- :func:`remove_duplicate_observables` -- deduplicates PennyLane observables by hash.
- :func:`build_sparse_arrays` -- converts :data:`CoeffTerms` into narrow, gathered
  NumPy arrays (no wasted width on untouched parameters).
- :func:`make_sparse_evaluator` -- compiles :data:`CoeffTerms` into fast numeric
  callables built on the narrow arrays. This is the only evaluator this fork
  keeps. It was measured ~6x faster than the removed dense ("standard")
  evaluator at typical k1/k2 truncation levels, and the removed JAX/vmap
  evaluator was consistently slower on CPU (see git history and the paper
  appendix for the old benchmarks that motivated dropping both). See
  ``notebooks/test/sparse_arrays_explained.ipynb`` for how the narrow
  representation works.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, List, Tuple

import numpy as np
from pennylane.operation import Observable

from ..pauli.sentence import CoeffTerms


def requires_propagation(method: Callable) -> Callable:
    """
    Decorator that guards a method behind a propagation check.

    Wraps any instance method so that it raises :exc:`RuntimeError` when called
    before :meth:`~pprop.propagator.Propagator.propagate` has been run (i.e.
    before ``self._propagated`` is ``True``).

    Parameters
    ----------
    method : Callable
        The instance method to wrap.

    Returns
    -------
    Callable
        The wrapped method with the propagation guard applied.

    Raises
    ------
    RuntimeError
        If ``self._propagated`` is ``False`` at call time.
    """
    def wrapper(self, *args, **kwargs):
        if not self._propagated:
            raise RuntimeError(
                f"You must call .propagate() before calling .{method.__name__}()"
            )
        return method(self, *args, **kwargs)
    return wrapper

def remove_duplicate_observables(
    observables: List[Observable],
) -> Tuple[List[Observable], List[Observable]]:
    """
    Remove duplicate observables from a list of PennyLane observables.

    Two observables are considered duplicates if their simplified canonical form
    has the same :attr:`~pennylane.operation.Operator.hash`. This avoids
    redundant propagations when an ansatz accidentally returns the same
    observable more than once.

    Parameters
    ----------
    observables : list[Observable]
        Raw list of PennyLane observables as captured from a
        :class:`~pennylane.tape.QuantumTape`.

    Returns
    -------
    unique_observables : list[Observable]
        Deduplicated list, each observable in its simplified canonical form.
    removed_elements : list[Observable]
        Observables that were dropped because an identical hash was already seen.
    """
    seen_hashes: set[int]         = set()
    unique_observables: List[Observable] = []
    removed_elements:  List[Observable] = []

    for tape_obs in observables:
        simplified = tape_obs.simplify()  # put into canonical form before hashing
        h = simplified.hash
        if h not in seen_hashes:
            unique_observables.append(simplified)
            seen_hashes.add(h)
        else:
            removed_elements.append(simplified)

    return unique_observables, removed_elements

def build_sparse_arrays(
    expr: CoeffTerms,
    num_params: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Convert a :data:`CoeffTerms` list into narrow, gathered NumPy arrays.

    Like :func:`build_arrays`, but instead of a full ``(n_terms, num_params)``
    row per term (mostly filled with the trivial power ``0``), only the
    parameters a term actually touches are stored, padded to a common width
    ``W`` = the largest number of distinct parameters touched by any single
    term in ``expr``. ``k1`` (Pauli weight truncation) bounds ``W`` directly,
    so ``W`` is typically far smaller than ``num_params`` for a heavily
    truncated propagation - evaluating the resulting arrays does the same
    computation as :func:`build_arrays`' output with much less wasted work
    (see ``notebooks/test/sparse_arrays_explained.ipynb`` for a worked example
    and measurements).

    Sin and cos are tracked with independent widths (``Ws``, ``Wc``), since a
    term's sine and cosine supports need not be the same size.

    Parameters
    ----------
    expr : CoeffTerms
        List of ``(coeff, sin_indices, cos_indices)`` tuples. Indices may repeat
        (encoding powers > 1).
    num_params : int
        Total number of circuit parameters (only used to size the fallback
        ``coeffs``-only case; the returned arrays never have a ``num_params``-sized
        axis).

    Returns
    -------
    coeffs : ndarray of shape (n_terms,), dtype float64
    idx_sin : ndarray of shape (n_terms, Ws), dtype int64
        Parameter index touched by each sin factor; padding entries are ``0``.
    pow_sin : ndarray of shape (n_terms, Ws), dtype float64
        Power of that sin factor; padding entries are ``0`` (making the padded
        factor ``sin(theta)**0 = 1``, a no-op, regardless of ``idx_sin``'s
        padding value).
    idx_cos, pow_cos : ndarray
        As ``idx_sin``/``pow_sin``, for the cosine factors.
    """
    packed = []
    for coeff, sin_idx, cos_idx in expr:
        packed.append((coeff, list(Counter(sin_idx).items()), list(Counter(cos_idx).items())))

    n = len(packed)
    Ws = max((len(s) for _, s, _ in packed), default=1) or 1
    Wc = max((len(c) for _, _, c in packed), default=1) or 1

    coeffs = np.zeros(n, dtype=np.float64)
    idx_sin = np.zeros((n, Ws), dtype=np.int64)
    pow_sin = np.zeros((n, Ws), dtype=np.float64)
    idx_cos = np.zeros((n, Wc), dtype=np.int64)
    pow_cos = np.zeros((n, Wc), dtype=np.float64)

    for i, (coeff, sin_items, cos_items) in enumerate(packed):
        coeffs[i] = coeff
        for j, (idx, p) in enumerate(sin_items):
            idx_sin[i, j], pow_sin[i, j] = idx, p
        for j, (idx, p) in enumerate(cos_items):
            idx_cos[i, j], pow_cos[i, j] = idx, p

    return coeffs, idx_sin, pow_sin, idx_cos, pow_cos


def make_sparse_evaluator(
    expr: CoeffTerms,
    num_params: int,
) -> Tuple[Callable[[np.ndarray, np.ndarray], float],
           Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]]:
    """
    Compile a :data:`CoeffTerms` expression into fast numeric callables, using
    the narrow/gathered representation from :func:`build_sparse_arrays` instead
    of :func:`build_arrays`' dense ``(n_terms, num_params)`` arrays.

    Same interface and same values/gradients as :func:`make_evaluator` (to
    floating-point precision) - this is a drop-in replacement, just faster when
    ``k1``/``k2`` truncation makes each term touch only a small fraction of
    ``num_params`` (see ``notebooks/test/sparse_arrays_explained.ipynb`` for
    measurements: ~6x faster at the scales tested there).

    Like :func:`make_evaluator`, the returned callables take precomputed
    ``sins = sin(theta)``/``coss = cos(theta)`` rather than ``theta`` -
    :meth:`Propagator.__call__`/:meth:`Propagator.eval_and_grad` compute those
    once per call and share them across every observable. This also removes a
    second, local redundancy specific to this function: the previous version
    computed ``sin(theta)``/``cos(theta)`` twice each per call (once gathered
    at ``idx_sin``/``idx_cos`` for the forward pass, again gathered at
    ``idx_sin``/``idx_cos`` for the gradient's ``cos_at_sin``/``sin_at_cos``
    terms) - with ``sins``/``coss`` passed in already, both uses are just
    array-index gathers into the same precomputed array, not fresh
    ``np.sin``/``np.cos`` calls.

    Parameters
    ----------
    expr : CoeffTerms
        Symbolic expression as a list of ``(coeff, sin_indices, cos_indices)``
        tuples. Indices may repeat to encode powers.
    num_params : int
        Total number of circuit parameters.

    Returns
    -------
    eval : Callable[[ndarray, ndarray], float]
    eval_grad : Callable[[ndarray, ndarray], Tuple[float, ndarray]]
    """
    coeffs, idx_sin, pow_sin, idx_cos, pow_cos = build_sparse_arrays(expr, num_params)

    def _eval(sins: np.ndarray, coss: np.ndarray) -> float:
        sin_g = sins[idx_sin]  # gather: (n_terms, Ws)
        cos_g = coss[idx_cos]  # gather: (n_terms, Wc)
        sin_pow = np.where(pow_sin > 0, sin_g ** pow_sin, 1.0)
        cos_pow = np.where(pow_cos > 0, cos_g ** pow_cos, 1.0)
        return float((coeffs * sin_pow.prod(axis=1) * cos_pow.prod(axis=1)).sum())

    def _eval_grad(sins: np.ndarray, coss: np.ndarray) -> Tuple[float, np.ndarray]:
        sin_g = sins[idx_sin]
        cos_g = coss[idx_cos]
        sin_pow = np.where(pow_sin > 0, sin_g ** pow_sin, 1.0)
        cos_pow = np.where(pow_cos > 0, cos_g ** pow_cos, 1.0)
        sin_prod, cos_prod = sin_pow.prod(axis=1), cos_pow.prod(axis=1)
        term_vals = coeffs * sin_prod * cos_prod

        def excl(pow_arr: np.ndarray) -> np.ndarray:
            # Prefix/suffix cumulative products - same trick as make_evaluator,
            # just over the narrow W axis instead of num_params.
            m = pow_arr.shape[0]
            left = np.cumprod(np.concatenate([np.ones((m, 1)), pow_arr[:, :-1]], axis=1), axis=1)
            right = np.cumprod(np.concatenate([pow_arr[:, 1:], np.ones((m, 1))], axis=1)[:, ::-1], axis=1)[:, ::-1]
            return left * right

        excl_sin, excl_cos = excl(sin_pow), excl(cos_pow)
        cos_at_sin, sin_at_cos = coss[idx_sin], sins[idx_cos]  # gather, not a fresh cos/sin call

        # Clamp the exponent rather than testing the base. Padding entries carry
        # pow == 0 and would otherwise evaluate x ** -1; clamping to >= 0 avoids
        # that just as well, and keeps 0.0 ** 0 == 1.0 -- which is the correct
        # factor for a power-1 term sitting at exactly sin(theta) == 0, where
        # the previous `sin_g != 0` guard returned 0.0 and zeroed a gradient
        # component that is not zero.
        d_sin = np.where(
            pow_sin > 0,
            pow_sin * sin_g ** np.maximum(pow_sin - 1.0, 0.0) * cos_at_sin,
            0.0,
        )
        d_cos = np.where(
            pow_cos > 0,
            -pow_cos * cos_g ** np.maximum(pow_cos - 1.0, 0.0) * sin_at_cos,
            0.0,
        )

        sin_grad_terms = coeffs[:, None] * d_sin * excl_sin * cos_prod[:, None]  # (n_terms, Ws)
        cos_grad_terms = coeffs[:, None] * sin_prod[:, None] * excl_cos * d_cos  # (n_terms, Wc)

        grad = np.zeros(num_params)
        np.add.at(grad, idx_sin.ravel(), sin_grad_terms.ravel())
        np.add.at(grad, idx_cos.ravel(), cos_grad_terms.ravel())
        return float(term_vals.sum()), grad

    return _eval, _eval_grad