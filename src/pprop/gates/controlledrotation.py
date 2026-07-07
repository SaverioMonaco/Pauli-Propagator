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
    **the user must pass** ``θ/2`` **as the parameter** for any ``CRX``,
    ``CRY``, or ``CRZ`` gate in the ansatz:

    .. code-block:: python

        # Correct: pass theta/2 so that pprop and PennyLane agree
        qml.CRX(params[i] / 2, wires=[0, 1])

        # Wrong: pprop will NOT match PennyLane
        qml.CRX(params[i], wires=[0, 1])

    The ``(1 \\pm \\cos\\theta)/2`` and ``\\sin(\\theta)/2`` factors (e.g.
    ``"IY"``, ``"ZZ"`` entries) are represented exactly in ``θ`` and require
    no rescaling.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from numpy import cos, integer, intp, prod, sin
from pennylane import CRX as qmlCRX
from pennylane import CRY as qmlCRY
from pennylane import CRZ as qmlCRZ

from ..pauli.op import PauliOp
from ..pauli.sentence import CoeffTerms, PauliDict
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

    def evolve(self, word: Tuple[PauliOp, CoeffTerms]) -> PauliDict:
        """
        Heisenberg-evolve a Pauli word through this controlled rotation gate.

        For each matching rule entry the existing :data:`CoeffTerms` are
        scaled by the rule's multiplier: the multiplier's ``sin_idx`` and
        ``cos_idx`` (which use ``-1`` as a placeholder) are substituted with
        ``self.parameter`` and then appended to every existing term's
        index lists.

        Parameters
        ----------
        word : tuple[PauliOp, CoeffTerms]
            ``(pauliop, coeff_terms)`` pair to evolve.

        Returns
        -------
        PauliDict
            Evolved Pauli words with updated :data:`CoeffTerms`.
        """
        op, coeff_terms = word
        wire0, wire1 = self.wires
        p = self.parameter

        rule = self.rule.get(op[wire0] + op[wire1], None)

        # Word commutes with the gate, pass through unchanged.
        if rule is None:
            return PauliDict({op: coeff_terms})

        evolved = PauliDict()

        for output_label, (m_coeff, m_sin, m_cos) in rule:
            # Build the output Pauli word.
            new_op = op.copy()
            new_op.set(wire0, output_label[0])
            new_op.set(wire1, output_label[1])

            # Substitute the placeholder -1 with the actual parameter index.
            sin_ext = [p if i == _P else i for i in m_sin]
            cos_ext = [p if i == _P else i for i in m_cos]

            # Scale each existing term by the multiplier and extend index lists.
            new_terms: CoeffTerms = []
            for c, s, cc in coeff_terms:
                if isinstance(self.parameter, (integer, intp)):
                    new_terms.append((
                        m_coeff * c,
                        list(s) + sin_ext,
                        list(cc) + cos_ext,
                    ))
                else:
                    new_terms.append((
                        m_coeff * c * prod(sin(sin_ext)) * prod(cos(cos_ext)),
                        list(s),
                        list(cc),
                    ))

            if new_terms:
                evolved.add_terms(new_op, new_terms)

        return evolved


# ---------------------------------------------------------------------------
# Concrete gates
# ---------------------------------------------------------------------------

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
        Pass ``params[i] / 2`` to ``qml.CRX`` to match.

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
        Pass ``params[i] / 2`` to ``qml.CRY`` to match.

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
        Pass ``params[i] / 2`` to ``qml.CRZ`` to match.

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