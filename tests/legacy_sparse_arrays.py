"""
Legacy padded/gathered array layout, kept only for the test suite.

This is ``build_sparse_arrays`` as it existed in ``pprop.propagator.evaluator``
before the ragged (CSR-style) layout replaced it (see
``pprop.propagator.evaluator.build_ragged_arrays``). It's no longer part of
the public API - it lives here purely so the test suite can still cross-check
the ragged evaluator's numbers against the older, simpler-but-wasteful
padded-array approach.
"""
from __future__ import annotations

from collections import Counter
from typing import Tuple

import numpy as np

from pprop.pauli.sentence import CoeffTerms


def build_sparse_arrays(
    expr: CoeffTerms,
    num_params: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Convert a :data:`CoeffTerms` list into narrow, gathered NumPy arrays.

    Instead of a full ``(n_terms, num_params)`` row per term (mostly filled
    with the trivial power ``0``), only the parameters a term actually
    touches are stored, padded to a common width ``W`` = the largest number
    of distinct parameters touched by any single term in ``expr``.

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


def eval_sparse_arrays(
    coeffs: np.ndarray,
    idx_sin: np.ndarray,
    pow_sin: np.ndarray,
    idx_cos: np.ndarray,
    pow_cos: np.ndarray,
    sins: np.ndarray,
    coss: np.ndarray,
) -> float:
    """Evaluate the padded arrays from :func:`build_sparse_arrays` directly."""
    sin_g = sins[idx_sin]
    cos_g = coss[idx_cos]
    sin_pow = np.where(pow_sin > 0, sin_g ** pow_sin, 1.0)
    cos_pow = np.where(pow_cos > 0, cos_g ** pow_cos, 1.0)
    return float((coeffs * sin_pow.prod(axis=1) * cos_pow.prod(axis=1)).sum())
