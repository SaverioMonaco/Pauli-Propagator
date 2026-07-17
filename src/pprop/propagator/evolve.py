"""This module handles the core evolution of Pauli words through a list of gates
via the Heisenberg picture.
"""
from typing import List, Tuple

from ..gates.base import Gate
from ..pauli.sentence import CoeffTerms, PauliDict
from .pruning import Pruner
from .truncation import Truncation


def to_expectation(paulidict: PauliDict) -> CoeffTerms:
    r"""
    Extract the expectation value expression from a propagated :class:`~pprop.pauli.sentence.PauliDict`.

    In the :math:`|0\rangle^{\otimes n}` computational basis state, only Pauli words
    composed entirely of :math:`Z` and :math:`I` operators have non-zero expectation:

    .. math::

        \langle 0 | I | 0 \rangle = 1, \quad
        \langle 0 | Z | 0 \rangle = 1, \quad
        \langle 0 | X | 0 \rangle = 0, \quad
        \langle 0 | Y | 0 \rangle = 0

    This function iterates over all Pauli words in ``paulidict``, keeps only
    those that satisfy the zero-bracket condition (i.e. only :math:`Z`/:math:`I`
    on every qubit), and concatenates their :data:`CoeffTerms` into a single
    flat list representing the full expectation value expression.

    Parameters
    ----------
    paulidict : PauliDict
        Mapping of ``PauliOp -> CoeffTerms`` after Heisenberg evolution.

    Returns
    -------
    CoeffTerms
        Flat list of :data:`CoeffTerm` tuples whose sum gives the expectation
        value :math:`\langle 0 | O | 0 \rangle`.
    """
    expr: CoeffTerms = []
    for pauliword, coeffterms in paulidict.items():
        # Only Z and I operators have non-zero expectation in the |0⟩ state.
        if pauliword.zerobracket():
            expr += coeffterms  # coeffterms is a CoeffTerms (list), so += extends the list
    return expr


def heisenberg(
    gates: List[Gate],
    paulidict: PauliDict,
    debug: bool = False,
    pruners: List[Pruner] = [],
    truncations: List[Truncation] = [],
) -> Tuple[PauliDict, CoeffTerms]:
    r"""
    Evolve a :class:`~pprop.pauli.sentence.PauliDict` backwards through a list of gates
    (Heisenberg picture).

    Each gate is applied in *reverse* order so that the observable is propagated
    from the measurement end of the circuit back to the input. After all gates
    have been applied, :func:`to_expectation` extracts the symbolic expectation
    value expression as a :data:`CoeffTerms` list.

    Parameters
    ----------
    gates : list[pprop.gates.Gate]
        Ordered list of gates as they appear in the circuit (will be iterated
        in reverse).
    paulidict : PauliDict
        Initial observable represented as a mapping of ``PauliOp -> CoeffTerms``.
    debug : bool, optional
        If ``True``, print the gate, pre-evolution, and post-evolution state at
        each step. Defaults to ``False``.
    pruners : list[Pruner], optional
        Exact pruning strategies applied before each gate step.
    truncations : list[Truncation], optional
        Approximate truncation strategies applied after each gate step.

    Returns
    -------
    paulidict : PauliDict
        The fully evolved observable after all gates have been applied.
    expectation : CoeffTerms
        Flat list of :data:`CoeffTerm` tuples encoding the symbolic expectation
        value :math:`\langle 0 | U^\dagger O U | 0 \rangle`.
    """
    reversed_gates = gates[::-1]
    history = []

    for pruner in pruners:
        pruner.setup(reversed_gates)

    for truncation in truncations:
        truncation.setup(reversed_gates)

    for i, gate in enumerate(reversed_gates):
        # A gate whose wires don't touch any qubit currently carrying a
        # non-identity operator in `paulidict` acts as the identity on every
        # term present - skip pruning/truncation/evolution for it entirely.
        # This is exact, not approximate: pruners/truncations only ever base
        # their decisions on the *current* paulidict and the *remaining*
        # gates, both of which are unaffected by a gate that overlaps
        # nothing currently tracked, so deferring their next call to the
        # next relevant gate is indistinguishable from calling them here.
        # Skipped only when `debug=False`, so `debug=True` keeps its full
        # per-gate trace/history unchanged.
        if not debug and not (gate.wire_mask & paulidict.active_mask):
            continue

        pauli_add    = PauliDict()  # Evolved replacement terms to add
        pauli_remove = PauliDict()  # Original terms to remove after evolution

        for pruner in pruners:
            pruner.prune(paulidict, i)

        for pauliword, coeffterms in paulidict.items():
            # Evolve this (pauliword, coeffterms) pair through the gate.
            evolved: PauliDict = gate.evolve((pauliword, coeffterms))

            pauli_add    += evolved
            # Only the key (PauliOp) matters here; the coefficient is irrelevant
            # because we are removing the entire entry from paulidict.
            pauli_remove[pauliword] = []

        if debug:
            print("=== Evolve ===")
            print("GATE:", gate)
            print(" PRE:", paulidict)

        # Swap out the original terms for their evolved counterparts.
        paulidict -= pauli_remove
        paulidict += pauli_add

        if debug:
            print("  REM:", pauli_remove)
            print("  ADD:", pauli_add)
            print("POST:", paulidict)
            history.append(paulidict.copy())

        for truncation in truncations:
            truncation.truncate(paulidict, i)

    return paulidict, to_expectation(paulidict), history