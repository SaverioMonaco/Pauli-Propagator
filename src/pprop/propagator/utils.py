"""
Utility functions for the Propagator class.

Provides:

- :func:`requires_propagation` -- decorator guarding methods until propagation is done.
- :func:`remove_duplicate_observables` -- deduplicates PennyLane observables by hash.
- :func:`build_sparse_arrays` -- converts :data:`CoeffTerms` into narrow, gathered
  NumPy arrays (no wasted width on untouched parameters). See
  ``notebooks/test/sparse_arrays_explained.ipynb`` for how the narrow
  representation works.
- :func:`build_ragged_arrays` -- the same idea without the padding: sin and cos
  factors in one concatenated list, indexed against a single lookup table.
- :func:`make_sparse_evaluator` -- compiles :data:`CoeffTerms` into fast numeric
  callables built on the ragged arrays. This is the only evaluator
  this fork keeps. It was measured ~6x faster than the removed dense
  ("standard") evaluator at typical k1/k2 truncation levels, and the removed
  JAX/vmap evaluator was consistently slower on CPU (see git history and the
  paper appendix for the old benchmarks that motivated dropping both).
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


def build_ragged_arrays(
    expr: CoeffTerms,
    num_params: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Convert a :data:`CoeffTerms` list into a ragged (CSR-style) layout.

    :func:`build_sparse_arrays` pads every term out to ``W``, the largest number
    of distinct parameters touched by any single term, which wastes a share of
    every gather and product whenever the support sizes vary - and under
    ``k1``/``k2`` truncation they vary a lot. This function instead concatenates
    the terms' factors end to end and records how many belong to each term, so
    there is no padding at all.

    Sine and cosine factors go into the *same* list, indexing a single lookup
    table laid out as::

        [ sin(theta_0) ... sin(theta_{P-1}), 1.0, cos(theta_0) ... cos(theta_{P-1}) ]

    with ``P = num_params``: a sine factor on parameter ``k`` is index ``k``, a
    cosine factor is ``num_params + 1 + k``, and index ``num_params`` is a
    sentinel carrying the neutral value ``1``. One list means the evaluator
    gathers and reduces once per call rather than once for each of sin and cos.

    Powers stay encoded as repeated indices, exactly as in ``expr``, instead of
    being compressed to ``(index, power)`` pairs. Powers above ``1`` are rare,
    so the extra entries cost little, and in exchange the forward pass never
    calls ``**`` and the gradient never multiplies by a power.

    Parameters
    ----------
    expr : CoeffTerms
        List of ``(coeff, sin_indices, cos_indices)`` tuples. Indices may repeat
        (encoding powers > 1).
    num_params : int
        Total number of circuit parameters, including fixed-value slots.

    Returns
    -------
    coeffs : ndarray of shape (n_terms,), float64
    idx : ndarray of shape (nnz,), int64
        Table index of every factor, terms concatenated end to end.
    cnt : ndarray of shape (n_terms,), int64
        Number of factors belonging to each term.

    Notes
    -----
    A constant term, carrying no sin or cos factor at all, gets a single entry
    pointing at the sentinel, so that every run is non-empty and
    ``np.multiply.reduceat`` needs no special case.
    """
    sentinel = num_params
    cos_offset = num_params + 1

    coeff_out: list[float] = []
    idx: list[int] = []
    cnt: list[int] = []

    for coeff, sin_idx, cos_idx in expr:
        entries: list[int] = [j for j in sin_idx]
        entries += [cos_offset + j for j in cos_idx]
        if not entries:
            entries.append(sentinel)

        coeff_out.append(coeff)
        cnt.append(len(entries))
        idx.extend(entries)

    return (np.asarray(coeff_out, dtype=np.float64),
            np.asarray(idx, dtype=np.int64),
            np.asarray(cnt, dtype=np.int64))


def _make_ragged_evaluator(expr, num_params, tol):
    """Build the ``(eval, eval_grad)`` pair for one block of terms."""
    coeffs, idx, cnt = build_ragged_arrays(expr, num_params)

    n_terms = len(coeffs)
    if n_terms == 0:
        # An empty expression; nothing to evaluate.
        zero = np.zeros(num_params)
        return (lambda sins, coss: 0.0,
                lambda sins, coss: (0.0, zero.copy()))

    cos_offset = num_params + 1
    # Start offset of each term's run, for np.multiply.reduceat, and the
    # owning term of each entry, for broadcasting term values back out.
    off = np.zeros(n_terms, dtype=np.int64)
    np.cumsum(cnt[:-1], out=off[1:])
    row = np.repeat(np.arange(n_terms), cnt)
    # Parameters this block actually reads, for the near-singular check below.
    used = np.unique(idx)
    used_sin = used[used < num_params]
    used_cos = used[used >= cos_offset] - cos_offset

    def _table(sins: np.ndarray, coss: np.ndarray) -> np.ndarray:
        """The [sin..., 1.0, cos...] lookup table the entries index into."""
        table = np.empty(2 * num_params + 1)
        table[:num_params] = sins
        table[num_params] = 1.0
        table[cos_offset:] = coss
        return table

    def _eval(sins: np.ndarray, coss: np.ndarray) -> float:
        factors = _table(sins, coss)[idx]
        return float((coeffs * np.multiply.reduceat(factors, off)).sum())

    def _eval_grad(sins: np.ndarray, coss: np.ndarray) -> Tuple[float, np.ndarray]:
        factors = _table(sins, coss)[idx]
        term_vals = coeffs * np.multiply.reduceat(factors, off)

        # Differentiating one factor of a product gives the product itself
        # times that factor's logarithmic derivative:
        #     d(term)/d(theta_k) = term * cot(theta_k)   [sin factor]
        #                        = term * -tan(theta_k)  [cos factor]
        # which avoids building exclusive products. Neither depends on the
        # entry, only on the parameter it reads, so the entries just have to
        # accumulate term values per parameter and the cot/tan multiply is one
        # pass over num_params rather than over every factor. A power p is p
        # repeated entries, which accumulate p*cot on their own.
        acc = np.bincount(idx, term_vals[row], 2 * num_params + 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cot = coss / sins
            tan = sins / coss
        # cot/tan blow up at zeros of sin/cos: zero them everywhere they are
        # singular (so no inf reaches the product) and recompute the affected
        # parameters exactly below.
        cot[np.abs(sins) < tol] = 0.0
        tan[np.abs(coss) < tol] = 0.0
        grad = cot * acc[:num_params] - tan * acc[cos_offset:]

        # Exact recomputation for those parameters: rebuild the per-term
        # product *excluding* the offending factor, one extra reduceat pass
        # each. There is usually nothing in these loops.
        for k in used_sin[np.abs(sins[used_sin]) < tol]:
            hit = idx == k
            rows, p = np.unique(row[hit], return_counts=True)
            without_k = factors.copy(); without_k[hit] = 1.0
            rest = np.multiply.reduceat(without_k, off)[rows]
            grad[k] += np.dot(coeffs[rows] * rest, p * sins[k] ** (p - 1) * coss[k])
        for k in used_cos[np.abs(coss[used_cos]) < tol]:
            hit = idx == cos_offset + k
            rows, q = np.unique(row[hit], return_counts=True)
            without_k = factors.copy(); without_k[hit] = 1.0
            rest = np.multiply.reduceat(without_k, off)[rows]
            grad[k] += np.dot(coeffs[rows] * rest, -q * coss[k] ** (q - 1) * sins[k])

        return float(term_vals.sum()), grad

    return _eval, _eval_grad


def make_sparse_evaluator(
    expr: CoeffTerms,
    num_params: int,
    tol: float = 1e-6,
) -> Tuple[Callable[[np.ndarray, np.ndarray], float],
           Callable[[np.ndarray, np.ndarray], Tuple[float, np.ndarray]]]:
    """
    Compile a :data:`CoeffTerms` expression into fast numeric callables, using
    the ragged representation from :func:`build_ragged_arrays` instead of
    :func:`build_arrays`' dense ``(n_terms, num_params)`` arrays.

    Same interface and same values/gradients as :func:`make_evaluator` (to
    floating-point precision) - this is a drop-in replacement, just faster when
    ``k1``/``k2`` truncation makes each term touch only a small fraction of
    ``num_params``. Nothing is padded, sine and cosine factors share one gather
    and one :func:`numpy.multiply.reduceat` pass, and the gradient reduces to a
    single :func:`numpy.bincount` over the factors followed by one multiply
    over the parameter vector.

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
    tol : float, optional
        A parameter whose ``|sin|`` or ``|cos|`` falls below this gets the exact
        (slower) gradient path instead of the cot/tan form, which is singular
        there. Only speed depends on this, not correctness. Defaults to ``1e-6``.

    Returns
    -------
    eval : Callable[[ndarray, ndarray], float]
    eval_grad : Callable[[ndarray, ndarray], Tuple[float, ndarray]]
    """
    return _make_ragged_evaluator(expr, num_params, tol)
