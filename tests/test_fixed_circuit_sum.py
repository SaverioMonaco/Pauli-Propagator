"""Circuits without trainable gates: each Pauli word's terms are summed into one."""
import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator

N, L = 6, 5
_ANGLES = np.random.default_rng(0).uniform(-np.pi, np.pi, (L, N, 2))


def _ansatz(trainable_last_layer=False):
    def ansatz(params):
        for layer in range(L):
            for q in range(N):
                for axis, gate in enumerate((qml.RX, qml.RY)):
                    if trainable_last_layer and layer == L - 1:
                        gate(params[2 * q + axis], wires=q)
                    else:
                        gate(float(_ANGLES[layer, q, axis]), wires=q)
            for q in range(layer % 2, N - 1, 2):
                qml.CNOT(wires=[q, q + 1])
        return [qml.expval(qml.PauliZ(N // 2)), qml.expval(qml.PauliX(0) @ qml.PauliY(1))]
    return ansatz


def _propagate(n_threads=1, **kw):
    p = Propagator(_ansatz(**kw))
    p.propagate(n_threads=n_threads)
    return p


@pytest.mark.parametrize("n_threads", [1, 4])
def test_fixed_circuit_matches_statevector(n_threads):
    p = _propagate(n_threads)
    assert p.num_params == 0
    ref = np.array(qml.QNode(_ansatz(), qml.device("default.qubit", wires=N))(None))
    np.testing.assert_allclose(np.ravel(p(np.array([]))), ref, atol=1e-12)


def test_fixed_circuit_one_term_per_word():
    # Every surviving term is a plain number, at most one per Pauli word: far
    # fewer than the paths, and never more than the 2**N diagonal words.
    for expr in _propagate().exprs:
        assert all(not s and not c for _, s, c in expr)
        assert len(expr) <= 2 ** N


def test_fixed_circuit_independent_of_threads():
    assert _propagate(1).exprs == _propagate(4).exprs


def test_trainable_circuit_not_summed():
    # One trainable layer turns the summing off; the values are still exact.
    p = _propagate(trainable_last_layer=True)
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, p.num_params)
    ref = np.array(qml.QNode(_ansatz(True), qml.device("default.qubit", wires=N))(theta))
    np.testing.assert_allclose(np.ravel(p(theta)), ref, atol=1e-12)
