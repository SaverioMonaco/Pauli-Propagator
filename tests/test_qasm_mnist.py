"""Tests for pprop.qasm (export) and the offline parts of pprop.mnist."""
import re

import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator, mnist
from pprop.qasm import to_qasm

PL_GATES = {"h": qml.Hadamard, "s": qml.S, "sx": qml.SX, "t": qml.T,
            "rx": qml.RX, "ry": qml.RY, "rz": qml.RZ,
            "cx": qml.CNOT, "cy": qml.CY, "cz": qml.CZ,
            "crx": qml.CRX, "cry": qml.CRY, "crz": qml.CRZ, "swap": qml.SWAP}


def gates(params):
    qml.Hadamard(0); qml.S(1); qml.SX(2); qml.T(0)
    qml.RX(params[0], wires=0); qml.RY(params[1], wires=1); qml.RZ(params[2], wires=2)
    qml.CNOT(wires=[0, 1]); qml.CY(wires=[1, 2]); qml.CZ(wires=[2, 0])
    qml.CRX(params[3], wires=[0, 2]); qml.CRY(params[4], wires=[2, 1]); qml.CRZ(params[5], wires=[1, 0])
    qml.SWAP(wires=[0, 2]); qml.RY(0.7, wires=1)  # fixed angle too


def ansatz(params):
    gates(params)
    return qml.expval(qml.PauliZ(0))


def replay(qasm: str):
    """Queue the gates of a QASM string as PennyLane ops."""
    for line in qasm.splitlines():
        m = re.match(r"(\w+)(?:\(([^)]*)\))? (q\[.*);", line)
        if m and m[1] in PL_GATES:
            angles = [float(a) for a in m[2].split(",")] if m[2] else []
            PL_GATES[m[1]](*angles, wires=[int(w) for w in re.findall(r"q\[(\d+)\]", m[3])])


def test_qasm_reproduces_state():
    params = np.random.default_rng(0).normal(size=6)
    dev = qml.device("default.qubit", wires=3)
    expected = qml.QNode(lambda: (gates(params), qml.state())[1], dev)()
    got = qml.QNode(lambda: (replay(to_qasm(ansatz, params)), qml.state())[1], dev)()
    assert np.allclose(got, expected)


def test_qasm_accepts_propagator_and_header():
    prop = Propagator(ansatz)
    qasm = to_qasm(prop, np.zeros(prop.num_params), measure=False)
    assert qasm.startswith('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[3];\n')
    assert "measure" not in qasm and "creg" not in qasm


def test_qasm_unknown_gate():
    with pytest.raises(ValueError, match="Toffoli"):
        to_qasm(lambda p: qml.Toffoli(wires=[0, 1, 2]), [])


def test_mnist_moments_and_bitstrings():
    X = np.array([[0, 1, 1], [1, 1, 0]])
    assert np.allclose(mnist.z_moments(X, [(0,), (1,), (0, 2)]), [0.0, -1.0, -1.0])
    # Qiskit order: qubit 0 is the rightmost character
    assert np.array_equal(mnist.bitstrings_to_array(["110", "011"]), [[0, 1, 1], [1, 1, 0]])
    assert mnist.mmd(X, X, sigma=1.0) == pytest.approx(0.0, abs=1e-12)
    assert mnist.as_images(np.zeros((2, 9))).shape == (2, 3, 3)
