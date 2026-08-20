"""
Utility functions for the Propagator class.

Provides:

- :func:`requires_propagation` -- decorator guarding methods until propagation is done.
- :func:`remove_duplicate_observables` -- deduplicates PennyLane observables by hash.

The evaluator-building functions (``build_sparse_arrays``, ``build_ragged_arrays``,
``make_sparse_evaluator``) live in :mod:`pprop.propagator.evaluator`.
"""
from __future__ import annotations

from typing import Callable, List, Tuple

from pennylane.operation import Observable


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
