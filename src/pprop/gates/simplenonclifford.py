"""
This submodule defines :class:`SimpleNonClifford`, the base class for
single-qubit non-parametrised non-Clifford gates, and the concrete gate
:class:`T`.
"""
from math import sqrt
from typing import Dict, List, Optional, Tuple

from pennylane import T as qmlT

from ..pauli.op import PauliOp
from ..pauli.sentence import CoeffTerms, PauliDict
from .base import Gate

# Rule type: maps a single-qubit Pauli label to two (output_label, phase) pairs.
# e.g. "X" -> (("X", +1/√2), ("Y", -1/√2))
EvolutionRule = Dict[str, Tuple[Tuple[str, float], Tuple[str, float]]]


class SimpleNonClifford(Gate):
    """
    Base class for single-qubit non-parametrised non-Clifford gates.

    Unlike Clifford gates, which map every Pauli word to a single Pauli word,
    non-Clifford gates map one Pauli word to a *superposition* of two Pauli
    words with constant float coefficients. The specific mapping is defined
    by a ``rule`` dict supplied by each subclass.

    Parameters
    ----------
    wires : list[int]
        Qubits on which the gate acts (single-qubit gate, so one wire).
    qml_gate : pennylane.operation.Operator
        Corresponding PennyLane gate class, used for circuit drawing.
    parameter : int or None
        Non-Clifford gates are non-parametrised, so this is always ``None``.
    rule : EvolutionRule
        Dict mapping a single-qubit Pauli label (``"X"``, ``"Y"``, or
        ``"Z"``) to a pair of ``(output_label, phase)`` tuples describing
        the Heisenberg evolution of that Pauli through the gate. Labels
        absent from the dict commute with the gate and pass through unchanged.

    Attributes
    ----------
    rule : EvolutionRule
        The Heisenberg evolution rule for this gate.
    """

    def __init__(
        self,
        wires,
        qml_gate,
        parameter,
        rule: EvolutionRule,
    ) -> None:
        super().__init__(wires=wires, qml_gate=qml_gate, parameter=parameter)
        self.rule = rule

    def evolve(self, word: Tuple[PauliOp, CoeffTerms]) -> PauliDict:
        """
        Heisenberg-evolve a Pauli word through this non-Clifford gate.

        Looks up the single-qubit Pauli at the gate's wire in ``self.rule``.
        If no rule exists for that label the Pauli word commutes with the gate
        and is returned unchanged. Otherwise the word is split into two new
        Pauli words whose scalar coefficients are scaled by the corresponding
        constant phases.

        Parameters
        ----------
        word : tuple[PauliOp, CoeffTerms]
            ``(pauliop, coeff_terms)`` pair to evolve.

        Returns
        -------
        PauliDict
            A :class:`~pprop.pauli.sentence.PauliDict` with either one entry
            (if the word commutes with the gate) or two entries (if it does not).
        """
        op, coeff_terms = word
        wire = self.wires[0]
        rule = self.rule.get(op[wire], None)

        # If no rule exists this Pauli commutes with the gate, pass through unchanged.
        if rule is None:
            return PauliDict({op: coeff_terms})

        (basis1, phase1), (basis2, phase2) = rule

        # Build the two output Pauli words by updating the qubit at `wire`.
        new_op1 = op.copy()
        new_op1.set(wire, basis1)

        new_op2 = op.copy()
        new_op2.set(wire, basis2)

        # Scale every existing CoeffTerm by the constant phase factor.
        # sin/cos index lists are copied to avoid mutating the original.
        scaled1: CoeffTerms = [(phase1 * c, list(s), list(cc)) for c, s, cc in coeff_terms]
        scaled2: CoeffTerms = [(phase2 * c, list(s), list(cc)) for c, s, cc in coeff_terms]

        return PauliDict({new_op1: scaled1, new_op2: scaled2})


class T(SimpleNonClifford):
    r"""
    The single-qubit T gate.

    .. math::

        T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\pi/4} \end{bmatrix}

    The Heisenberg evolution rules are:

    .. math::

        X \;\mapsto\; \tfrac{1}{\sqrt{2}} X - \tfrac{1}{\sqrt{2}} Y

        Y \;\mapsto\; \tfrac{1}{\sqrt{2}} Y + \tfrac{1}{\sqrt{2}} X

        Z \;\mapsto\; Z

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : float, int, optional
        Unused for non-parametrised gates. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "X": (("X", +1 / sqrt(2)), ("Y", -1 / sqrt(2))),
            "Y": (("Y", +1 / sqrt(2)), ("X", +1 / sqrt(2))),
            # Z commutes with T, no rule needed, handled by the base class fallthrough.
        }
        super().__init__(wires, qmlT, parameter, rule)