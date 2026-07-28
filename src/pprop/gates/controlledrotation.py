"""
This submodule defines :class:`ControlledRotationGate`, the base class for
single-parameter controlled rotation gates, and the concrete gates
:class:`CRX`, :class:`CRY`, and :class:`CRZ`.

Coefficient encoding
--------------------
Controlled rotation gates produce factors of the form
:math:`\\cos(\\theta/2)`, :math:`\\sin(\\theta/2)`,
:math:`\\cos^2(\\theta/2)`, :math:`\\sin^2(\\theta/2)`, and
:math:`\\sin(\\theta/2)\\cos(\\theta/2)`.  Setting :math:`p` =
``parameter``, these map directly onto :data:`~pprop.pauli.sentence.CoeffTerm`
tuples with repeated indices:

.. list-table::
   :header-rows: 1

   * - Factor
     - CoeffTerm multiplier
   * - :math:`\\cos(\\theta/2)`
     - ``(1.0, [], [p])``
   * - :math:`\\sin(\\theta/2)`
     - ``(1.0, [p], [])``
   * - :math:`\\cos^2(\\theta/2) = (1+\\cos\\theta)/2`
     - ``(1.0, [], [p, p])``
   * - :math:`\\sin^2(\\theta/2) = (1-\\cos\\theta)/2`
     - ``(1.0, [p, p], [])``
   * - :math:`\\sin(\\theta/2)\\cos(\\theta/2) = \\sin(\\theta)/2`
     - ``(1.0, [p], [p])``


.. warning::

    **Half-angle convention for** ``cos(θ/2)`` **and** ``sin(θ/2)`` **terms.**

    Rules involving ``cos(θ/2)`` or ``sin(θ/2)`` (e.g. the ``"XI"``, ``"YI"``
    entries) cannot be represented exactly as :data:`~pprop.pauli.sentence.CoeffTerm`
    tuples in ``θ``, only in ``θ/2``.  To match PennyLane's output exactly,
    write the ansatz normally, with the gate reading the same trainable index
    as any other rotation, and instead **halve that index's entry in the
    array you pass to** :meth:`~pprop.propagator.Propagator.__call__` **or**
    :meth:`~pprop.propagator.Propagator.eval_and_grad`:

    .. code-block:: python

        # Ansatz: write it exactly like a normal PennyLane circuit.
        def ansatz(params):
            qml.CRX(params[0], wires=[0, 1])
            ...

        prop = Propagator(ansatz)
        prop.propagate()

        # Correct: halve theta at eval time, not inside the ansatz.
        prop(np.array([theta / 2, ...]))

        # Wrong: passing theta unmodified will NOT match PennyLane's
        # qml.CRX(theta, wires=[0, 1]).
        prop(np.array([theta, ...]))

    Do **not** write ``qml.CRX(params[i] / 2, wires=...)`` in the ansatz
    itself: since ``Propagator`` captures gate parameters by their *type*
    (``int``/``np.integer`` means trainable index, ``float`` means a fixed,
    non-trainable value, see :class:`~pprop.gates.base.Gate`), dividing the
    placeholder index by 2 during capture turns it into a fixed float instead
    of halving the trainable value at evaluation time.

    The ``(1 \\pm \\cos\\theta)/2`` and ``\\sin(\\theta)/2`` factors (e.g.
    ``"IY"``, ``"ZZ"`` entries) are represented exactly in ``θ`` and require
    no rescaling, but are still read from the same halved slot for uniformity.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from pennylane import CRX as qmlCRX
from pennylane import CRY as qmlCRY
from pennylane import CRZ as qmlCRZ

from .base import Gate

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

# Each rule entry maps a two-character Pauli string to a list of
# (output_label_pair, CoeffTerm_multiplier) pairs.
# The multiplier is a single CoeffTerm (c, sin_idx, cos_idx) expressed in
# terms of the gate's parameter index p; the actual index is substituted
# at evolve-time.
# We store the sin/cos index lists as relative placeholders (using -1) and
# replace -1 with self.parameter inside evolve().
_RuleEntry = List[Tuple[str, Tuple[float, List[int], List[int]]]]
EvolutionRule = Dict[str, _RuleEntry]

# Sentinel value used as a placeholder for parameter in the rule dicts.
_P = -1


class ControlledRotationGate(Gate):
    """
    Base class for single-parameter two-qubit controlled rotation gates.

    Unlike :class:`~pprop.gates.rotation_gate.RotationGate`, controlled
    rotations act non-trivially only when the control qubit is in the
    :math:`|1\\rangle` state.  This produces factors of
    :math:`\\cos(\\theta/2)`, :math:`\\sin(\\theta/2)`, and their squares,
    each encoded as a :data:`~pprop.pauli.sentence.CoeffTerm` with repeated
    ``parameter`` entries (see module docstring for the full table).

    Each rule entry maps a two-character Pauli string ``"PQ"``
    (control ⊗ target) to a list of ``(output_label, multiplier)`` pairs,
    where ``multiplier`` is a :data:`~pprop.pauli.sentence.CoeffTerm` with
    ``-1`` as a placeholder for ``parameter``.

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    qml_gate : pennylane.operation.Operation
        Corresponding PennyLane gate class.
    parameter : int, float
        Index of :math:`\\theta` in the global parameter vector if int.
        Actual value of the rotation if float.
    rule : EvolutionRule
        Heisenberg evolution rule dict.

    Attributes
    ----------
    rule : EvolutionRule
        The evolution rule for this gate.
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


# ---------------------------------------------------------------------------
# Concrete gates
# ---------------------------------------------------------------------------
#
# These rule tables are the human-readable reference for CRX/CRY/CRZ's
# Heisenberg evolution; the Rust extension pprop_rs (crx_rule/cry_rule/
# crz_rule in native/pprop_rs/src/lib.rs) is what actually executes them
# during Propagator.propagate(), using -1 -> 0/1/2 repetitions of the
# parameter index in place of the _P placeholder substitution the removed
# evolve() method used to do here.

class CRX(ControlledRotationGate):
    r"""
    The controlled-:math:`R_x` gate.

    .. math::

        CR_x(\theta) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \cos(\theta) & -i\sin(\theta) \\
            0 & 0 & -i\sin(\theta) & \cos(\theta)
        \end{bmatrix}

    .. note::
        The parameter ``θ`` here corresponds to ``θ/2`` in PennyLane's convention.
        Write ``qml.CRX(params[i], wires=...)`` normally in the ansatz, and pass
        ``params[i] / 2`` (not the ansatz) to ``Propagator.__call__``/
        ``eval_and_grad`` to match PennyLane's ``qml.CRX(theta, ...)``.

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    parameter : int
        Index of :math:`\\theta` in the global parameter vector.
    """

    def __init__(self, wires: List[int], parameter: int) -> None:
        # Multiplier encoding (using _P as placeholder for parameter):
        #   cos(t/2)       -> (1.0, [],       [_P])
        #   sin(t/2)       -> (1.0, [_P],     [])
        #  -sin(t/2)       -> (-1.0,[_P],     [])
        #   cos²(t/2)      -> (1.0, [],       [_P, _P])
        #   sin²(t/2)      -> (1.0, [_P, _P], [])
        #   sin(t/2)cos(t/2) -> (1.0,[_P],   [_P])
        #  -sin(t/2)cos(t/2) -> (-1.0,[_P],  [_P])
        rule: EvolutionRule = {
            "IY": [("IY", (1.0,  [],       [_P, _P])),
                   ("IZ", (-1.0, [_P],     [_P])),
                   ("ZY", (1.0,  [_P, _P], [])),
                   ("ZZ", (1.0,  [_P],     [_P]))],
            "IZ": [("IZ", (1.0,  [],       [_P, _P])),
                   ("IY", (1.0,  [_P],     [_P])),
                   ("ZZ", (1.0,  [_P, _P], [])),
                   ("ZY", (-1.0, [_P],     [_P]))],
            "XI": [("XI", (1.0,  [],       [_P])),
                   ("YX", (1.0,  [_P],     []))],
            "XX": [("XX", (1.0,  [],       [_P])),
                   ("YI", (1.0,  [_P],     []))],
            "XY": [("XY", (1.0,  [],       [_P])),
                   ("XZ", (-1.0, [_P],     []))],
            "XZ": [("XZ", (1.0,  [],       [_P])),
                   ("XY", (1.0,  [_P],     []))],
            "YI": [("YI", (1.0,  [],       [_P])),
                   ("XX", (-1.0, [_P],     []))],
            "YX": [("YX", (1.0,  [],       [_P])),
                   ("XI", (-1.0, [_P],     []))],
            "YY": [("YY", (1.0,  [],       [_P])),
                   ("YZ", (-1.0, [_P],     []))],
            "YZ": [("YZ", (1.0,  [],       [_P])),
                   ("YY", (1.0,  [_P],     []))],
            "ZY": [("ZY", (1.0,  [],       [_P, _P])),
                   ("ZZ", (-1.0, [_P],     [_P])),
                   ("IY", (1.0,  [_P, _P], [])),
                   ("IZ", (1.0,  [_P],     [_P]))],
            "ZZ": [("ZZ", (1.0,  [],       [_P, _P])),
                   ("ZY", (1.0,  [_P],     [_P])),
                   ("IZ", (1.0,  [_P, _P], [])),
                   ("IY", (-1.0, [_P],     [_P]))],
        }
        super().__init__(wires, qmlCRX, parameter, rule)


class CRY(ControlledRotationGate):
    r"""
    The controlled-:math:`R_y` gate.

    .. math::

        CR_y(\theta) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & \cos(\theta) & -\sin(\theta) \\
            0 & 0 & \sin(\theta) & \cos(\theta)
        \end{bmatrix}

    .. note::
        The parameter ``θ`` here corresponds to ``θ/2`` in PennyLane's convention.
        Write ``qml.CRY(params[i], wires=...)`` normally in the ansatz, and pass
        ``params[i] / 2`` (not the ansatz) to ``Propagator.__call__``/
        ``eval_and_grad`` to match PennyLane's ``qml.CRY(theta, ...)``.

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    parameter : int
        Index of :math:`\\theta` in the global parameter vector.
    """

    def __init__(self, wires: List[int], parameter: int) -> None:
        rule: EvolutionRule = {
            "IX": [("IX", (1.0,  [],       [_P, _P])),
                   ("IZ", (1.0,  [_P],     [_P])),
                   ("ZX", (1.0,  [_P, _P], [])),
                   ("ZZ", (-1.0, [_P],     [_P]))],
            "IZ": [("IZ", (1.0,  [],       [_P, _P])),
                   ("IX", (-1.0, [_P],     [_P])),
                   ("ZZ", (1.0,  [_P, _P], [])),
                   ("ZX", (1.0,  [_P],     [_P]))],
            "XI": [("XI", (1.0,  [],       [_P])),
                   ("YY", (1.0,  [_P],     []))],
            "XX": [("XX", (1.0,  [],       [_P])),
                   ("XZ", (1.0,  [_P],     []))],
            "XY": [("XY", (1.0,  [],       [_P])),
                   ("YI", (1.0,  [_P],     []))],
            "XZ": [("XZ", (1.0,  [],       [_P])),
                   ("XX", (-1.0, [_P],     []))],
            "YI": [("YI", (1.0,  [],       [_P])),
                   ("XY", (-1.0, [_P],     []))],
            "YX": [("YX", (1.0,  [],       [_P])),
                   ("YZ", (1.0,  [_P],     []))],
            "YY": [("YY", (1.0,  [],       [_P])),
                   ("XI", (-1.0, [_P],     []))],
            "YZ": [("YZ", (1.0,  [],       [_P])),
                   ("YX", (-1.0, [_P],     []))],
            "ZX": [("ZX", (1.0,  [],       [_P, _P])),
                   ("ZZ", (1.0,  [_P],     [_P])),
                   ("IX", (1.0,  [_P, _P], [])),
                   ("IZ", (-1.0, [_P],     [_P]))],
            "ZZ": [("ZZ", (1.0,  [],       [_P, _P])),
                   ("ZX", (-1.0, [_P],     [_P])),
                   ("IZ", (1.0,  [_P, _P], [])),
                   ("IX", (1.0,  [_P],     [_P]))],
        }
        super().__init__(wires, qmlCRY, parameter, rule)


class CRZ(ControlledRotationGate):
    r"""
    The controlled-:math:`R_z` gate.

    .. math::

        CR_z(\theta) = \begin{bmatrix}
            1 & 0 & 0 & 0 \\
            0 & 1 & 0 & 0 \\
            0 & 0 & e^{-i\theta} & 0 \\
            0 & 0 & 0 & e^{i\theta}
        \end{bmatrix}

    .. note::
        The parameter ``θ`` here corresponds to ``θ/2`` in PennyLane's convention.
        Write ``qml.CRZ(params[i], wires=...)`` normally in the ansatz, and pass
        ``params[i] / 2`` (not the ansatz) to ``Propagator.__call__``/
        ``eval_and_grad`` to match PennyLane's ``qml.CRZ(theta, ...)``.

    Parameters
    ----------
    wires : list[int]
        ``[control, target]`` qubit indices.
    parameter : int
        Index of :math:`\\theta` in the global parameter vector.
    """

    def __init__(self, wires: List[int], parameter: int) -> None:
        rule: EvolutionRule = {
            "IX": [("IX", (1.0,  [],       [_P, _P])),
                   ("IY", (-1.0, [_P],     [_P])),
                   ("ZX", (1.0,  [_P, _P], [])),
                   ("ZY", (1.0,  [_P],     [_P]))],
            "IY": [("IY", (1.0,  [],       [_P, _P])),
                   ("IX", (1.0,  [_P],     [_P])),
                   ("ZY", (1.0,  [_P, _P], [])),
                   ("ZX", (-1.0, [_P],     [_P]))],
            "XI": [("XI", (1.0,  [],       [_P])),
                   ("YZ", (1.0,  [_P],     []))],
            "XX": [("XX", (1.0,  [],       [_P])),
                   ("XY", (-1.0, [_P],     []))],
            "XY": [("XY", (1.0,  [],       [_P])),
                   ("XX", (1.0,  [_P],     []))],
            "XZ": [("XZ", (1.0,  [],       [_P])),
                   ("YI", (1.0,  [_P],     []))],
            "YI": [("YI", (1.0,  [],       [_P])),
                   ("XZ", (-1.0, [_P],     []))],
            "YX": [("YX", (1.0,  [],       [_P])),
                   ("YY", (-1.0, [_P],     []))],
            "YY": [("YY", (1.0,  [],       [_P])),
                   ("YX", (1.0,  [_P],     []))],
            "YZ": [("YZ", (1.0,  [],       [_P])),
                   ("XI", (-1.0, [_P],     []))],
            "ZX": [("ZX", (1.0,  [],       [_P, _P])),
                   ("ZY", (-1.0, [_P],     [_P])),
                   ("IX", (1.0,  [_P, _P], [])),
                   ("IY", (1.0,  [_P],     [_P]))],
            "ZY": [("ZY", (1.0,  [],       [_P, _P])),
                   ("ZX", (1.0,  [_P],     [_P])),
                   ("IY", (1.0,  [_P, _P], [])),
                   ("IX", (-1.0, [_P],     [_P]))],
        }
        super().__init__(wires, qmlCRZ, parameter, rule)