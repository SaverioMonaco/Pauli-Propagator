"""
This submodule defines :class:`ControlledGate`, the base class for two-qubit
non-parametrised controlled gates, and the concrete gates :class:`CNOT`,
:class:`CY`, and :class:`CZ`.
"""
from typing import Dict, List, Optional, Tuple

from pennylane import CNOT as qmlCNOT
from pennylane import CY as qmlCY
from pennylane import CZ as qmlCZ

from ..pauli.op import PauliOp
from ..pauli.sentence import CoeffTerms, PauliDict
from .base import Gate

# Rule type: maps a two-character Pauli string (control + target) to
# ((output_control, output_target), sign).
# e.g. "IY" -> (("Z", "Y"), +1)
EvolutionRule = Dict[str, Tuple[Tuple[str, str], int]]


class ControlledGate(Gate):
    """
    Base class for two-qubit non-parametrised controlled gates.

    The Heisenberg evolution rule is encoded as a dict keyed by two-character
    strings ``"PQ"`` where ``P`` is the Pauli at the control wire and ``Q`` is
    the Pauli at the target wire. Each entry maps to an
    ``((output_control, output_target), sign)`` tuple. Two-qubit Pauli
    combinations absent from the dict commute with the gate and pass through
    unchanged.

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    qml_gate : pennylane.operation.Operator
        Corresponding PennyLane gate class, used for circuit drawing.
    parameter : int or None
        Index into the parameter vector. Controlled gates are non-parametrised,
        so this is always ``None``.
    rule : EvolutionRule
        Dict mapping a two-character Pauli string (e.g. ``"IY"``) to a
        ``((output_control, output_target), sign)`` pair.

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
        Heisenberg-evolve a Pauli word through this controlled gate.

        Looks up the two-character Pauli string ``op[control] + op[target]``
        in ``self.rule``. If absent the word commutes with the gate and is
        returned unchanged. Otherwise both qubits are updated and all scalar
        coefficients are multiplied by the sign.

        Parameters
        ----------
        word : tuple[PauliOp, CoeffTerms]
            ``(pauliop, coeff_terms)`` pair to evolve.

        Returns
        -------
        PauliDict
            One entry (the evolved word).
        """
        op, coeff_terms = word
        wire0, wire1 = self.wires

        # Look up the two-qubit Pauli combination at (control, target).
        rule = self.rule.get(op[wire0] + op[wire1], None)

        # If no rule exists this Pauli commutes with the gate, pass through unchanged.
        if rule is None:
            return PauliDict({op: coeff_terms})

        (out0, out1), sign = rule

        new_op = op.copy()
        new_op.set(wire0, out0)
        new_op.set(wire1, out1)

        # Scale coefficients by the sign; avoid unnecessary list comprehension for +1.
        if sign == 1:
            new_terms = list(coeff_terms)
        else:
            new_terms: CoeffTerms = [(sign * c, s, cc) for c, s, cc in coeff_terms]

        return PauliDict({new_op: new_terms})


class CNOT(ControlledGate):
    r"""
    The Controlled-NOT (CX) gate.

    .. math::

        \text{CNOT} = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & 0 & 1 \\
            0 & 0 & 1 & 0
        \end{bmatrix}

    The Heisenberg evolution maps each two-qubit Pauli string
    ``control ⊗ target`` according to the rule dict. All other combinations
    commute with the gate.

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    parameter : float, int, optional
        Unused. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "IY": (("Z", "Y"), +1),
            "IZ": (("Z", "Z"), +1),
            "XI": (("X", "X"), +1),
            "XX": (("X", "I"), +1),
            "XY": (("Y", "Z"), +1),
            "XZ": (("Y", "Y"), -1),
            "YI": (("Y", "X"), +1),
            "YX": (("Y", "I"), +1),
            "YY": (("X", "Z"), -1),
            "YZ": (("X", "Y"), +1),
            "ZY": (("I", "Y"), +1),
            "ZZ": (("I", "Z"), +1),
        }
        super().__init__(wires, qmlCNOT, parameter, rule)


class CY(ControlledGate):
    r"""
    The Controlled-Y gate.

    .. math::

        \text{CY} = \begin{bmatrix}
            1 & 0 & 0 &  0 \\
            0 & 1 & 0 &  0 \\
            0 & 0 & 0 & -i \\
            0 & 0 & i &  0
        \end{bmatrix}

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    parameter : float, int, optional
        Unused. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "IX": (("Z", "X"), +1),
            "IZ": (("Z", "Z"), +1),
            "XI": (("X", "Y"), +1),
            "XX": (("Y", "Z"), -1),
            "XY": (("X", "I"), +1),
            "XZ": (("Y", "X"), +1),
            "YI": (("Y", "Y"), +1),
            "YX": (("X", "Z"), +1),
            "YY": (("Y", "I"), +1),
            "YZ": (("X", "X"), -1),
            "ZX": (("I", "X"), +1),
            "ZZ": (("I", "Z"), +1),
        }
        super().__init__(wires, qmlCY, parameter, rule)


class CZ(ControlledGate):
    r"""
    The Controlled-Z gate.

    .. math::

        \text{CZ} = \begin{bmatrix}
            1 & 0 & 0 &  0 \\
            0 & 1 & 0 &  0 \\
            0 & 0 & 1 &  0 \\
            0 & 0 & 0 & -1
        \end{bmatrix}

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    parameter : float, int, optional
        Unused. Defaults to ``None``.
    """

    def __init__(self, wires: List[int], parameter: Optional[int] = None) -> None:
        rule: EvolutionRule = {
            "IX": (("Z", "X"), +1),
            "IY": (("Z", "Y"), +1),
            "XI": (("X", "Z"), +1),
            "XX": (("Y", "Y"), +1),
            "XY": (("Y", "X"), -1),
            "XZ": (("X", "I"), +1),
            "YI": (("Y", "Z"), +1),
            "YX": (("X", "Y"), -1),
            "YY": (("X", "X"), +1),
            "YZ": (("Y", "I"), +1),
            "ZX": (("I", "X"), +1),
            "ZY": (("I", "Y"), +1),
        }
        super().__init__(wires, qmlCZ, parameter, rule)