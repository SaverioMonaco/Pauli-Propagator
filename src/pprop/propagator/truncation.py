"""
Truncation strategies for Heisenberg-picture Pauli propagation.

Overview
--------
Unlike :mod:`~pprop.propagator.pruning` strategies, which are *exact* (they
only discard terms that are provably zero), truncation strategies are
*approximate*: they drop terms that are small or that exceed a given budget,
introducing a controlled approximation error.

This submodule provides:

* :class:`Truncation` abstract base class defining the interface.
* :class:`WeightTruncation` removes Pauli words whose weight (number of
  non-identity operators) exceeds a given threshold after each gate step.
* :class:`FrequencyTruncation` removes individual coefficient terms whose
  total trigonometric frequency (number of sin/cos factors) exceeds a
  threshold after each gate step.
* :class:`CoefficientTruncation` removes individual coefficient terms whose
  scalar magnitude falls below a threshold.

All truncations share the same two-phase lifecycle as
:class:`~pprop.propagator.pruning.Pruner`:

1. **Setup** (:meth:`Truncation.setup`) called *once* before the evolution
   loop with the full list of gates in reversed order.

2. **Truncate** (:meth:`Truncation.truncate`) called *once per gate step*,
   just after the gate is applied, with the current
   :class:`~pprop.pauli.sentence.PauliDict` and the index of the current step.
   Dead/small entries are removed or filtered in-place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..gates.base import Gate
from ..pauli.sentence import PauliDict


class Truncation(ABC):
    """
    Abstract base class for approximate Heisenberg-evolution truncation strategies.

    A :class:`Truncation` drops terms that exceed a budget or fall below a
    magnitude threshold, trading exactness for a smaller :class:`~pprop.pauli.sentence.PauliDict`.

    The contract is:

    * :meth:`setup` is called exactly *once*, before the evolution loop
      starts, with ``reversed_gates``.

    * :meth:`truncate` is called at the *end* of each loop iteration,
      after the gate at position ``step`` has been applied.  It must
      remove or filter entries from ``paulidict`` in-place.

    .. warning::

        Truncated propagation is **not** exact.  The resulting expectation
        value is an approximation to the true Heisenberg-evolved expectation.
    """

    @abstractmethod
    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Precompute auxiliary data for the full gate sequence.

        Called once before the evolution loop.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in the order they will be consumed during Heisenberg
            evolution (i.e. the original circuit gates reversed).
        """
        ...

    @abstractmethod
    def truncate(self, paulidict: PauliDict, step: int) -> None:
        """
        Remove or filter approximated entries from *paulidict* in-place.

        Called at the end of each loop iteration, after the gate at
        index ``step`` in ``reversed_gates`` has been applied.

        Parameters
        ----------
        paulidict : PauliDict
            The live observable terms at the current evolution step.
            Modified in-place.
        step : int
            Index of the gate that was just applied (0-based, indexing
            into ``reversed_gates`` as passed to :meth:`setup`).
        """
        ...


class WeightTruncation(Truncation):
    """
    Truncate Pauli words whose weight exceeds a given threshold.

    After each gate step, any Pauli word with weight (number of non-identity
    single-qubit operators) strictly greater than ``threshold`` is discarded.

    Parameters
    ----------
    threshold : int
        Maximum allowed Pauli weight. Words with weight > ``threshold``
        are removed.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def setup(self, reversed_gates: List[Gate]) -> None:
        pass  # no precomputation needed

    def truncate(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete words with weight strictly above the threshold.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step. Modified in-place.
        step : int
            Index of the gate that was just applied.
        """
        threshold = self.threshold
        dead_keys = [pw for pw, _ in paulidict.items() if pw.weight() > threshold]
        dead = PauliDict({pw: [] for pw in dead_keys})
        paulidict.remove_keys_from_dict(dead)


class FrequencyTruncation(Truncation):
    """
    Truncate coefficient terms whose trigonometric frequency exceeds a threshold.

    After each gate step, any :data:`CoeffTerm` ``(c, sin_idx, cos_idx)`` with
    ``len(sin_idx) + len(cos_idx) > threshold`` is discarded.  A word is
    removed entirely when all of its terms have been filtered out.

    Parameters
    ----------
    threshold : int
        Maximum allowed trigonometric frequency (total number of sin/cos
        factors). Terms with frequency > ``threshold`` are removed.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold

    def setup(self, reversed_gates: List[Gate]) -> None:
        pass  # no precomputation needed

    def truncate(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete individual coefficient terms above the frequency threshold;
        remove words that become empty as a result.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step. Modified in-place.
        step : int
            Index of the gate that was just applied.
        """
        threshold = self.threshold
        dead_keys = []
        for pw, terms in paulidict.items():
            filtered = [t for t in terms if len(t[1]) + len(t[2]) <= threshold]
            if not filtered:
                dead_keys.append(pw)
            elif len(filtered) < len(terms):
                paulidict[pw] = filtered

        dead = PauliDict({pw: [] for pw in dead_keys})
        paulidict.remove_keys_from_dict(dead)


class CoefficientTruncation(Truncation):
    """
    Truncate individual coefficient terms whose scalar magnitude falls below a threshold.

    For each Pauli word, any :data:`CoeffTerm` ``(c, sin_idx, cos_idx)`` with
    ``|c| < threshold`` is discarded.  A word is removed entirely when all of
    its terms have been filtered out.

    Parameters
    ----------
    threshold : float
        Minimum scalar coefficient magnitude to retain. Default is ``1e-10``.
    """

    def __init__(self, threshold: float = 1e-10) -> None:
        self.threshold = threshold

    def setup(self, reversed_gates: List[Gate]) -> None:
        pass  # no precomputation needed

    def truncate(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete individual coefficient terms below the threshold; remove words
        that become empty as a result.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step. Modified in-place.
        step : int
            Index of the gate that was just applied.
        """
        threshold = self.threshold
        dead_keys = []
        for pw, terms in paulidict.items():
            filtered = [t for t in terms if abs(t[0]) >= threshold]
            if not filtered:
                dead_keys.append(pw)
            elif len(filtered) < len(terms):
                paulidict[pw] = filtered

        dead = PauliDict({pw: [] for pw in dead_keys})
        paulidict.remove_keys_from_dict(dead)
