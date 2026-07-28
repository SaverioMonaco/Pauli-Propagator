"""
Cross-checks every gate's Python ``rule`` dict (``pprop/gates/*.py``) against
the Rust extension's rule tables (``rx_rule``/``h_rule``/``cnot_rule``/...
in ``native/pprop_rs/src/lib.rs``) via ``pprop_rs.evolve_single_gate_debug``.

Why this exists: the Python ``rule`` dicts are never executed at propagation
time (that's all done by their Rust counterparts) and were, until this test
existed, never actually checked against the Rust tables by anything
automated - only by a one-time manual pass during the Python-to-Rust port
(see ``personal/rust_port.tex``). Nothing stopped `lib.rs` and the matching
``pprop/gates/*.py`` dict from silently drifting apart. This test makes that
claim true: for every explicit entry in every gate's ``rule`` dict, it builds
the single input Pauli word that entry describes, evolves it through the
*actual* Rust gate implementation, and asserts the result matches what the
dict says should happen, term-for-term.

Labels absent from a ``rule`` dict (the "commutes, passes through
unchanged" case) aren't exercised here - that behavior is structural (a
``match`` arm's ``_ => None`` fallthrough, not per-gate data) and is already
covered by the full random-circuit checks in ``test_backends.py``.
"""
import math

import pprop_rs
import pytest

from pprop.gates.controlled import CNOT, CY, CZ
from pprop.gates.controlledrotation import CRX, CRY, CRZ
from pprop.gates.rotation import RX, RY, RZ
from pprop.gates.simpleclifford import H, S, SWAP, SX
from pprop.gates.simplenonclifford import T
from pprop.propagator import _GATE_KIND

# PauliOp's bitmask convention (see pprop/pauli/op.py): bit0 = x-bit, bit1 = z-bit.
_LABEL_TO_XZ = {"I": (0, 0), "X": (1, 0), "Z": (0, 1), "Y": (1, 1)}

# Arbitrary, non-default parameter index - anything works, since these tests
# only check *how many times* it gets pushed onto sin_idx/cos_idx, not its
# numeric value.
_PARAM = 7


def _word(labels: str, wires: tuple[int, ...]) -> tuple[int, int]:
    """Build the (x, z) bitmask pair for `labels[i]` on `wires[i]`."""
    x = z = 0
    for label, wire in zip(labels, wires):
        xb, zb = _LABEL_TO_XZ[label]
        x |= xb << wire
        z |= zb << wire
    return x, z


def _evolve(kind_name: str, wire0: int, wire1: int, param: int, x: int, z: int):
    num_qubits = max(wire0, wire1) + 1
    return pprop_rs.evolve_single_gate_debug(
        num_qubits, _GATE_KIND[kind_name], wire0, wire1, param,
        [x], [z], [(1.0, [], [])],
    )


def _assert_matches(actual, expected):
    """
    Assert `actual` (raw pprop_rs.evolve_single_gate_debug output) and
    `expected` (list of (x, z, coeff, sin_idx, cos_idx)) contain the same
    rows, order-independent, coefficients compared with float tolerance.
    """
    remaining = list(actual)
    for ex_x, ex_z, ex_c, ex_s, ex_c_idx in expected:
        for i, (ax, az, ac, as_, ac_idx) in enumerate(remaining):
            if (
                ax == [ex_x] and az == [ex_z]
                and as_ == ex_s and ac_idx == ex_c_idx
                and math.isclose(ac, ex_c, rel_tol=1e-9, abs_tol=1e-12)
            ):
                del remaining[i]
                break
        else:
            raise AssertionError(
                f"expected row (x={ex_x}, z={ex_z}, c={ex_c}, sin={ex_s}, cos={ex_c_idx}) "
                f"not found in actual output {actual}"
            )
    assert not remaining, f"actual output has unexpected extra rows: {remaining}"


# --------------------------------------------------------------------- #
# RX/RY/RZ: label -> (out_label, sign). Splits into a cos(param) branch
# (same word) and a sign*sin(param) branch (out_label word).
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [RX, RY, RZ])
def test_rotation_rule_matches_rust(cls):
    gate = cls(wires=[0], parameter=_PARAM)
    kind_name = gate.qml_gate.name
    for label, (out_label, sign) in gate.rule.items():
        x, z = _word(label, (0,))
        out_x, out_z = _word(out_label, (0,))
        actual = _evolve(kind_name, 0, -1, _PARAM, x, z)
        expected = [
            (x, z, 1.0, [], [_PARAM]),
            (out_x, out_z, float(sign), [_PARAM], []),
        ]
        _assert_matches(actual, expected)


# --------------------------------------------------------------------- #
# H/S/SX: label -> (out_label, sign). Exactly one output word, constant sign.
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [H, S, SX])
def test_clifford1q_rule_matches_rust(cls):
    gate = cls(wires=[0])
    kind_name = gate.qml_gate.name
    for label, (out_label, sign) in gate.rule.items():
        x, z = _word(label, (0,))
        out_x, out_z = _word(out_label, (0,))
        actual = _evolve(kind_name, 0, -1, -1, x, z)
        expected = [(out_x, out_z, float(sign), [], [])]
        _assert_matches(actual, expected)


# --------------------------------------------------------------------- #
# T: label -> ((basis1, phase1), (basis2, phase2)). Splits into two words
# with constant (non-trig) phase multipliers.
# --------------------------------------------------------------------- #
def test_t_rule_matches_rust():
    gate = T(wires=[0])
    kind_name = gate.qml_gate.name
    for label, ((basis1, phase1), (basis2, phase2)) in gate.rule.items():
        x, z = _word(label, (0,))
        x1, z1 = _word(basis1, (0,))
        x2, z2 = _word(basis2, (0,))
        actual = _evolve(kind_name, 0, -1, -1, x, z)
        expected = [
            (x1, z1, float(phase1), [], []),
            (x2, z2, float(phase2), [], []),
        ]
        _assert_matches(actual, expected)


# --------------------------------------------------------------------- #
# SWAP has no rule dict (its evolution is a direct label exchange, not a
# lookup table) - check its stated behaviour directly across every
# (label0, label1) pair instead of walking a `.rule` attribute.
# --------------------------------------------------------------------- #
def test_swap_matches_rust():
    gate = SWAP(wires=[0, 1])
    kind_name = gate.qml_gate.name
    labels = "IXYZ"
    for label0 in labels:
        for label1 in labels:
            x, z = _word(label0 + label1, (0, 1))
            out_x, out_z = _word(label1 + label0, (0, 1))
            actual = _evolve(kind_name, 0, 1, -1, x, z)
            expected = [(out_x, out_z, 1.0, [], [])]
            _assert_matches(actual, expected)


# --------------------------------------------------------------------- #
# CNOT/CY/CZ: "PQ" -> ((out_control, out_target), sign). Exactly one output
# word, constant sign.
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [CNOT, CY, CZ])
def test_controlled_rule_matches_rust(cls):
    gate = cls(wires=[0, 1])
    kind_name = gate.qml_gate.name
    for pq, ((out_c, out_t), sign) in gate.rule.items():
        x, z = _word(pq, (0, 1))
        out_x, out_z = _word(out_c + out_t, (0, 1))
        actual = _evolve(kind_name, 0, 1, -1, x, z)
        expected = [(out_x, out_z, float(sign), [], [])]
        _assert_matches(actual, expected)


# --------------------------------------------------------------------- #
# CRX/CRY/CRZ: "PQ" -> list of (out_label_pair, (coeff, sin_placeholders,
# cos_placeholders)). Up to 4 output words, each scaling the input term by a
# constant coefficient and appending 0-2 copies of the parameter index to
# sin_idx/cos_idx.
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [CRX, CRY, CRZ])
def test_controlled_rotation_rule_matches_rust(cls):
    gate = cls(wires=[0, 1], parameter=_PARAM)
    kind_name = gate.qml_gate.name
    for pq, branches in gate.rule.items():
        x, z = _word(pq, (0, 1))
        actual = _evolve(kind_name, 0, 1, _PARAM, x, z)
        expected = []
        for out_label_pair, (coeff, sin_placeholders, cos_placeholders) in branches:
            out_x, out_z = _word(out_label_pair, (0, 1))
            expected.append((
                out_x, out_z, float(coeff),
                [_PARAM] * len(sin_placeholders),
                [_PARAM] * len(cos_placeholders),
            ))
        _assert_matches(actual, expected)
