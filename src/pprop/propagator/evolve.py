"""This module handles the core evolution of Pauli words through a list of gates
via the Heisenberg picture.
"""
from typing import List, Optional, Tuple

from ..gates.base import Gate
from ..pauli.sentence import CoeffTerms, PauliDict
from .pruning import Pruner


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
    gates : List[Gate],
    paulidict: PauliDict,
    k1: Optional[int],
    k2: Optional[int],
    debug: bool = False,
    pruners: List[Pruner] = [],
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
    k1 : int or None
        Pauli weight cutoff. Evolved terms whose Pauli weight exceeds ``k1``
        are discarded. ``None`` disables this truncation.
    k2 : int or None
        Frequency cutoff. Evolved terms whose total trigonometric frequency
        exceeds ``k2`` are discarded. ``None`` disables this truncation.
    opt : bool, optional
        If ``True``, use optimized pruning strategy. Defaults to ``False``.
    debug : bool, optional
        If ``True``, print the gate, pre-evolution, and post-evolution state at
        each step. Defaults to ``False``.

    Returns
    -------
    paulidict : PauliDict
        The fully evolved observable after all gates have been applied.
    expectation : CoeffTerms
        Flat list of :data:`CoeffTerm` tuples encoding the symbolic expectation
        value :math:`\langle 0 | U^\dagger O U | 0 \rangle`.
    """
    reversed_gates = gates[::-1]

    for pruner in pruners:
        pruner.setup(reversed_gates)
        
    for i, gate in enumerate(reversed_gates):
        pauli_add    = PauliDict()  # Evolved replacement terms to add
        pauli_remove = PauliDict()  # Original terms to remove after evolution

        for pauliword, coeffterms in paulidict.items():
            # Evolve this (pauliword, coeffterms) pair through the gate.
            evolved: PauliDict = gate.evolve((pauliword, coeffterms), k1, k2)

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

        for pruner in pruners:
            pruner.prune(paulidict, i)

    return paulidict, to_expectation(paulidict)