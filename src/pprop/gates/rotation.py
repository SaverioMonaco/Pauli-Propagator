"""
This submodule defines :class:`RotationGate`, the base class for single-qubit
parametrised Pauli rotation gates, and the concrete gates :class:`RX`,
:class:`RY`, and :class:`RZ`.

The ``rule`` tables below are the human-readable reference for these gates'
Heisenberg evolution; the Rust extension ``pprop_rs`` (``rx_rule``/``ry_rule``/
``rz_rule`` in ``native/pprop_rs/src/lib.rs``) is what actually executes them
during :meth:`~pprop.propagator.Propagator.propagate`.
"""
from typing import Dict, List, Tuple

from pennylane import RX as qmlRX
from pennylane import RY as qmlRY
from pennylane import RZ as qmlRZ

from .base import Gate

# Rule type: maps a single-qubit Pauli label to (output_label, sign).
# Absent labels commute with the rotation axis and pass through unchanged.
EvolutionRule = Dict[str, Tuple[str, int]]


class RotationGate(Gate):
    """
    Base class for single-qubit parametrised Pauli rotation gates.

    A rotation gate :math:`R_P(\\theta) = e^{-i\\theta P/2}` conjugates a
    Pauli word :math:`Q` according to:

    .. math::

        R_P^\\dagger\\, Q\\, R_P =
        \\begin{cases}
            Q & \\text{if } [Q, P] = 0 \\\\
            \\cos(\\theta)\\, Q + \\sigma \\sin(\\theta)\\, Q'
              & \\text{if } \\{Q, P\\} = 0
        \\end{cases}

    where :math:`Q'` is the Pauli obtained by applying the gate rule and
    :math:`\\sigma \\in \\{+1, -1\\}` is the sign given by the commutation
    relation. The trigonometric factors are encoded by appending the parameter
    index to the ``cos_idx`` or ``sin_idx`` lists of each :data:`CoeffTerm`.

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    qml_gate : pennylane.operation.Operator
        Corresponding PennyLane gate class, used for circuit drawing.
    parameter : float, int
        Index of :math:`\\theta` in the global parameter vector if int.
        Actual value of the rotation if float.
    rule : EvolutionRule
        Dict mapping a single-qubit Pauli label (``"X"``, ``"Y"``, or ``"Z"``)
        to a ``(output_label, sign)`` tuple for Paulis that anti-commute with
        the rotation axis. Labels absent from the dict commute and pass through
        unchanged.

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


class RX(RotationGate):
    r"""
    The single-qubit parametrised X rotation gate.

    .. math::

        R_x(\phi) = e^{-i\phi\,\sigma_x/2}

    Heisenberg evolution rules:

    .. math::

        Y \mapsto -\sin(\phi)\,Z + \cos(\phi)\,Y, \quad
        Z \mapsto +\sin(\phi)\,Y + \cos(\phi)\,Z, \quad
        X \mapsto X

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : int
        Index of :math:`\phi` in the global parameter vector.
    """

    def __init__(self, wires: List[int], parameter: int) -> None:
        rule: EvolutionRule = {
            "Y": ("Z", -1),
            "Z": ("Y", +1),
            # X commutes with RX, no rule needed.
        }
        super().__init__(wires, qmlRX, parameter, rule)


class RY(RotationGate):
    r"""
    The single-qubit parametrised Y rotation gate.

    .. math::

        R_y(\phi) = e^{-i\phi\,\sigma_y/2}

    Heisenberg evolution rules:

    .. math::

        X \mapsto +\sin(\phi)\,Z + \cos(\phi)\,X, \quad
        Z \mapsto -\sin(\phi)\,X + \cos(\phi)\,Z, \quad
        Y \mapsto Y

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : int
        Index of :math:`\phi` in the global parameter vector.
    """

    def __init__(self, wires: List[int], parameter: int) -> None:
        rule: EvolutionRule = {
            "X": ("Z", +1),
            "Z": ("X", -1),
            # Y commutes with RY, no rule needed.
        }
        super().__init__(wires, qmlRY, parameter, rule)


class RZ(RotationGate):
    r"""
    The single-qubit parametrised Z rotation gate.

    .. math::

        R_z(\phi) = e^{-i\phi\,\sigma_z/2}

    Heisenberg evolution rules:

    .. math::

        X \mapsto -\sin(\phi)\,Y + \cos(\phi)\,X, \quad
        Y \mapsto +\sin(\phi)\,X + \cos(\phi)\,Y, \quad
        Z \mapsto Z

    Parameters
    ----------
    wires : list[int]
        Qubit on which the gate acts.
    parameter : int
        Index of :math:`\phi` in the global parameter vector.
    """

    def __init__(self, wires: List[int], parameter: int) -> None:
        rule: EvolutionRule = {
            "X": ("Y", -1),
            "Y": ("X", +1),
            # Z commutes with RZ, no rule needed.
        }
        super().__init__(wires, qmlRZ, parameter, rule)