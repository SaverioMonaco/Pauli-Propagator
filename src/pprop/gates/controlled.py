"""
This submodule defines :class:`ControlledGate`, the base class for two-qubit
non-parametrised controlled gates, and the concrete gates :class:`CNOT`,
:class:`CY`, and :class:`CZ`.

The ``rule`` tables below are the human-readable reference for these gates'
Heisenberg evolution; the Rust extension ``pprop_rs`` (``cnot_rule``/``cy_rule``/
``cz_rule`` in ``native/pprop_rs/src/lib.rs``) is what actually executes them
during :meth:`~pprop.propagator.Propagator.propagate`.
"""
from typing import Dict, List, Optional, Tuple

from pennylane import CNOT as qmlCNOT
from pennylane import CY as qmlCY
from pennylane import CZ as qmlCZ

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