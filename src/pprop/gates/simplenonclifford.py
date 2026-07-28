"""
This submodule defines :class:`SimpleNonClifford`, the base class for
single-qubit non-parametrised non-Clifford gates, and the concrete gate
:class:`T`.

The ``rule`` table below is the human-readable reference for T's Heisenberg
evolution; the Rust extension ``pprop_rs`` (``t_rule`` in
``native/pprop_rs/src/lib.rs``) is what actually executes it during
:meth:`~pprop.propagator.Propagator.propagate`.
"""
from math import sqrt
from typing import Dict, List, Optional, Tuple

from pennylane import T as qmlT

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