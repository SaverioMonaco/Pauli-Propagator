"""
This submodule defines :class:`SimpleClifford`, the base class for single-qubit
Clifford gates, and the concrete gates :class:`H` and :class:`S`.

The ``rule`` tables below (and :class:`SWAP`'s evolution logic) are the
human-readable reference for these gates' Heisenberg evolution; the Rust
extension ``pprop_rs`` (``h_rule``/``s_rule``/``sx_rule``/``evolve_swap`` in
``native/pprop_rs/src/lib.rs``) is what actually executes them during
:meth:`~pprop.propagator.Propagator.propagate`.
"""
from typing import Dict, List, Optional, Tuple

from pennylane import SX as qmlSX
from pennylane import Hadamard as qmlH
from pennylane import S as qmlS

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

class SWAP(Gate):
    """
    The two-qubit SWAP gate.

    Heisenberg evolution rule: exchanges the Pauli labels on the two wires.

        P_i ⊗ Q_j  →  Q_i ⊗ P_j

    No sign change ever occurs.
    """

    def __init__(self, wires: List[int], parameter=None) -> None:
        from pennylane import SWAP as qmlSWAP
        assert len(wires) == 2, "SWAP requires exactly two wires."
        super().__init__(wires=wires, qml_gate=qmlSWAP, parameter=parameter)