"""
This submodule defines :class:`SimpleClifford`, the base class for single-qubit
Clifford gates, and the concrete gates :class:`H` and :class:`S`.
"""
from typing import Dict, List, Optional, Tuple

from pennylane import Hadamard as qmlH
from pennylane import S as qmlS
from pennylane import SX as qmlSX

from ..pauli.op import PauliOp
from ..pauli.sentence import CoeffTerms, PauliDict
from .base import Gate

# Rule type: maps a single-qubit Pauli label to (output_label, sign).
# e.g. "X" -> ("Z", +1) means X is mapped to +Z under conjugation.
EvolutionRule = Dict[str, Tuple[str, int]]


class SimpleClifford(Gate):
    """
    Base class for single-qubit non-parametrised Clifford gates.

    Clifford gates map every Pauli word to exactly one Pauli word, possibly
    with a sign flip. This is captured by a ``rule`` dict that maps each
    single-qubit Pauli label to an ``(output_label, sign)`` pair. Labels
    absent from the dict commute with the gate and pass through unchanged.

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    qml_gate : pennylane.operation.Operator
        Corresponding PennyLane gate class, used for circuit drawing.
    parameter : int or None
        Clifford gates are non-parametrised, so this is always ``None``.
    rule : EvolutionRule
        Dict mapping a single-qubit Pauli label (``"X"``, ``"Y"``, or ``"Z"``)
        to a ``(output_label, sign)`` tuple where ``sign`` is ``+1`` or ``-1``.

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

    def evolve(self, word: Tuple[PauliOp, CoeffTerms], k1, k2) -> PauliDict:
        """
        Heisenberg-evolve a Pauli word through this Clifford gate.

        Looks up the single-qubit Pauli at the gate's wire in ``self.rule``.
        If no rule exists the word commutes with the gate and is returned
        unchanged. Otherwise the qubit at that wire is replaced with the
        output label and all scalar coefficients are multiplied by the sign.

        Parameters
        ----------
        word : tuple[PauliOp, CoeffTerms]
            ``(pauliop, coeff_terms)`` pair to evolve.
        k1 : int or None
            Pauli weight cutoff (unused, Clifford gates do not change weight).
        k2 : int or None
            Frequency cutoff (unused, Clifford gates do not change frequency).

        Returns
        -------
        PauliDict
            A :class:`~pprop.pauli.sentence.PauliDict` with exactly one entry.
        """
        op, coeff_terms = word
        wire = self.wires[0]
        rule = self.rule.get(op[wire], None)

        # If no rule exists this Pauli commutes with the gate, pass through unchanged.
        if rule is None:
            return PauliDict({op: coeff_terms})

        output_label, sign = rule

        new_op = op.copy()
        new_op.set(wire, output_label)

        # Avoid unnecessary list comprehension when the sign is +1.
        if sign == 1:
            new_terms = list(coeff_terms)
        else:
            new_terms: CoeffTerms = [(sign * c, s, cc) for c, s, cc in coeff_terms]

        return PauliDict({new_op: new_terms})


class H(SimpleClifford):
    r"""
    The single-qubit Hadamard gate.

    .. math::

        H = \frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1\\ 1 & -1\end{bmatrix}

    Heisenberg evolution rules:

    .. math::

        X \mapsto Z, \quad Y \mapsto -Y, \quad Z \mapsto X

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : float, int, optional
        Unused. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "X": ("Z", +1),
            "Y": ("Y", -1),
            "Z": ("X", +1),
        }
        super().__init__(wires, qmlH, parameter, rule)


#: Alias for :class:`H`.
Hadamard = H


class S(SimpleClifford):
    r"""
    The single-qubit phase gate.

    .. math::

        S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

    Heisenberg evolution rules:

    .. math::

        X \mapsto -Y, \quad Y \mapsto X, \quad Z \mapsto Z

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : float, int, optional
        Unused. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "X": ("Y", -1),
            "Y": ("X", +1),
            # Z commutes with S, no rule needed, handled by the base class fallthrough.
        }
        super().__init__(wires, qmlS, parameter, rule)


class SX(SimpleClifford):
    r"""
    The single-qubit Square-Root X gate.

    .. math::

        \mathrm{SX} = \frac{1}{2}\begin{bmatrix} 1+i & 1-i \\ 1-i & 1+i \end{bmatrix}

    Heisenberg evolution rules:

    .. math::

        X \mapsto X, \quad Y \mapsto -Z, \quad Z \mapsto Y

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : float, int, optional
        Unused. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "Y": ("Z", -1),
            "Z": ("Y", +1),
            # X commutes with SX (SX is a function of X), no rule needed.
        }
        super().__init__(wires, qmlSX, parameter, rule)