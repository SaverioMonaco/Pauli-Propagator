"""
Pruning strategies for Heisenberg-picture Pauli propagation.

Overview
--------
During Heisenberg evolution a :class:`~pprop.pauli.sentence.PauliDict` can
grow large as gates split each Pauli word into multiple branches.  Many of
these branches are guaranteed to contribute zero to the final expectation
value :math:`\\langle 0 | O | 0 \\rangle` and can be discarded early,
before they are evolved further and spawn even more branches.

This submodule provides a small framework for such *pruning strategies*:

* :class:`Pruner` abstract base class defining the interface.
* :class:`DeadQubitPruner` removes words that carry an ``X`` or ``Y``
  operator on a qubit that will never be touched by any remaining gate and
  therefore can never be driven back to the :math:`Z/I` subspace required
  for a non-zero expectation value.
* :class:`XYWeightPruner` removes words whose XY-weight (number of qubits
  carrying ``X`` or ``Y``) exceeds the maximum reduction achievable by all
  remaining gates in the causal cone of that word.  Since each gate reduces
  XY-weight by at most 1, a word with too high a weight simply cannot reach
  XY-weight 0 before the circuit ends.

All pruners share the same two-phase lifecycle:

1. **Setup** (:meth:`Pruner.setup`) called *once* before the evolution
   loop with the full list of gates in reversed order.  This is where each
   pruner precomputes any auxiliary data structures it needs (e.g. suffix
   sets of active qubits).

2. **Prune** (:meth:`Pruner.prune`) called *once per gate step*, just
   before the gate is applied, with the current :class:`~pprop.pauli.sentence.PauliDict`
   and the index of the current step.  Dead entries are removed in-place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..gates.base import Gate
from ..pauli.sentence import PauliDict


class Pruner(ABC):
    """
    Abstract base class for Heisenberg-evolution pruning strategies.

    A :class:`Pruner` encapsulates a single dead-term elimination heuristic.
    Subclasses must implement :meth:`setup` and :meth:`prune`.

    The contract is:

    * :meth:`setup` is called exactly *once*, before the evolution loop
      starts, with ``reversed_gates``, the circuit gates in the order they
      are consumed during Heisenberg propagation (i.e. reversed with respect
      to the original circuit order).  Implementations should use this call
      to precompute any per-step data they need.

    * :meth:`prune` is called at the *beginning* of each loop iteration,
      before the gate at position ``step`` is applied.  It must remove all
      entries from ``paulidict`` that are provably dead, i.e. whose evolved
      descendants can never contribute to the expectation value, and leave
      all other entries untouched.

    .. warning::

        With prunings, the Pauli words carried in ``paulidict`` during evolution
        are **not** the exact Heisenberg-evolved observable.
        They are pruned to retain only  those terms that can possibly survive the
        :func:`~pprop.pauli.sentence.to_expectation` zero-bracket
        filter (i.e. words composed entirely of :math:`Z` and :math:`I`).
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
    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Remove provably dead entries from *paulidict* in-place.

        Called at the start of each loop iteration, before the gate at
        index ``step`` in ``reversed_gates`` is applied.

        Parameters
        ----------
        paulidict : PauliDict
            The live observable terms at the current evolution step.
            Modified in-place: dead entries are deleted.
        step : int
            Index of the gate that is about to be applied (0-based, indexing
            into ``reversed_gates`` as passed to :meth:`setup`).
        """
        ...

class DeadQubitPruner(Pruner):
    """
    Prune Pauli words that have a frozen ``X`` or ``Y`` on an inactive qubit.

    Correctness argument
    --------------------
    A Pauli word contributes to :math:`\\langle 0 | O | 0 \\rangle` only if
    every qubit carries either :math:`Z` or :math:`I` after *all* remaining
    gates have been applied (see :func:`~pprop.pauli.sentence.to_expectation`).
    A gate can change the operator on a qubit **only if that qubit appears in
    the gate's wire list**.  Therefore, if a qubit currently carries ``X`` or
    ``Y`` and no remaining gate (including the gate about to be applied at
    the current step) touches that qubit, its operator is permanently frozen
    in the non-:math:`ZI` state, the word can never reach zero-bracket and
    can be safely discarded.
    """

    def __init__(self) -> None:
        # Populated by setup(); empty until then.
        self._active_qubits_from: List[set] = []

    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Build the suffix-union table of active qubits.

        Iterates over ``reversed_gates`` from right to left, accumulating the
        union of wire sets so that ``_active_qubits_from[i]`` contains every
        qubit touched by gates ``i, i+1, …, n-1``.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in Heisenberg traversal order (reversed circuit order).
        """
        n = len(reversed_gates)

        # Sentinel: no gates remain after the last step.
        self._active_qubits_from = [set() for _ in range(n + 1)]

        # Right-to-left pass: each entry inherits all qubits from the next
        # step and adds the wires of the current gate.
        for i in range(n - 1, -1, -1):
            self._active_qubits_from[i] = self._active_qubits_from[i + 1].copy()
            self._active_qubits_from[i].update(reversed_gates[i].wires)

    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete words with a frozen ``X`` or ``Y`` from *paulidict*.

        For each Pauli word, iterates over its non-identity support and
        checks whether any qubit carrying ``X`` or ``Y`` is absent from
        ``_active_qubits_from[step]``.  If so, the word is removed.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step.  Modified in-place.
        step : int
            Index of the gate about to be applied.
        """
        active = self._active_qubits_from[step]

        # Build a bitmask of all active qubits.
        active_mask = 0
        for q in active:
            active_mask |= (1 << q)

        # A word is dead if it has any X or Y on an inactive qubit.
        # X or Y iff the x-bit is set. Inactive qubits are those NOT in active_mask.
        dead = PauliDict({
            pw: [] for pw, _ in paulidict.items()
            if pw.x & ~active_mask
        })

        paulidict.remove_keys_from_dict(dead)

class XYWeightPruner(Pruner):
    """
    Prune Pauli words whose XY-weight exceeds the maximum reduction
    achievable by the remaining gates in their causal cone.

    Correctness argument
    --------------------
    A Pauli word can only contribute to the expectation value if its
    XY-weight reaches zero by the end of evolution.  Each gate can reduce
    the XY-weight by at most 1.  However, a gate can only contribute to
    that reduction if it overlaps with the word's current XY support
    (qubits carrying X or Y).  A gate that touches an XY qubit may also
    spread the XY support to its other wires (conservative upper bound),
    so subsequent gates touching those new qubits are also counted.

    This causal-cone budget is a tight upper bound: if it is smaller than
    the current XY-weight the word can never reach the :math:`ZI` subspace,
    no matter how the gates are applied, and the word can be safely discarded.
    """

    def __init__(self) -> None:
        self._reversed_gates: List[Gate] = []
        self._gate_masks: List[int] = []

    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Cache the reversed gate list and precompute wire bitmasks.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in Heisenberg traversal order (reversed circuit order).
        """
        self._reversed_gates = reversed_gates
        self._gate_masks = [
            sum(1 << w for w in gate.wires) for gate in reversed_gates
        ]

    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete words whose XY-weight exceeds their causal-cone budget.

        For each Pauli word the method walks the remaining gates
        (``step`` to end, inclusive; the gate at ``step`` has not yet
        been applied) and counts how many of them overlap with the word's
        expanding XY support.  If that count is less than the XY-weight
        the word is provably dead and removed.

        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step. Modified in-place.
        step : int
            Index of the gate about to be applied.
        """
        gate_masks = self._gate_masks
        n = len(gate_masks)

        dead_keys = []
        for pw, _ in paulidict.items():
            xy_weight = pw.x.bit_count()
            if xy_weight == 0:
                continue  # already Z/I, cannot be pruned

            # Walk remaining gates and accumulate the causal-cone budget.
            # xy_support expands conservatively: when a gate overlaps it,
            # all of that gate's wires join the support (XY can spread).
            xy_support = pw.x
            budget = 0
            for i in range(step, n):
                gm = gate_masks[i]
                if gm & xy_support:
                    budget += 1
                    xy_support |= gm      # conservative support expansion
                    if budget >= xy_weight:
                        break             # budget already sufficient

            if budget < xy_weight:
                dead_keys.append(pw)

        dead = PauliDict({pw: [] for pw in dead_keys})
        paulidict.remove_keys_from_dict(dead)

class IQPPruner(Pruner):
    def __init__(self) -> None:
        pass

    def setup(self, reversed_gates: List[Gate]) -> None:
        """
        Build the suffix-sum budget table.

        Parameters
        ----------
        reversed_gates : list[Gate]
            Gates in Heisenberg traversal order (reversed circuit order).
        """
        self.prune_step = []

        direction = True
        weight = 0
        for i, gate in enumerate(reversed_gates):
            if gate.qml_gate.name in ['H', 'Hadamard', 'Barrier']:
                continue
            elif gate.qml_gate.name in ['CNOT']:
                weight += 2*int(direction) - 1
            elif gate.qml_gate.name in ["RZ"]:
                direction = not direction
            else:
                raise ValueError(f"Not a valid IQP gate: {gate.qml_gate.name}")
            
            if gate.qml_gate.name in ['CNOT'] and weight == 0:
                self.prune_step.append(i)


    def prune(self, paulidict: PauliDict, step: int) -> None:
        """
        Delete words with an X.
    
        Parameters
        ----------
        paulidict : PauliDict
            Observable terms at the current step. Modified in-place.
        step : int
            Index of the gate about to be applied.
        """    
        # A word is dead if it has any X
        if step in self.prune_step:
            dead = PauliDict({
                pw: [] for pw, _ in paulidict.items()
                if not pw.zerobracket_X()
            })

            paulidict.remove_keys_from_dict(dead)