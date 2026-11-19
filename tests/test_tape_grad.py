"""Gradients of tapes (TapeProgram.eval_and_grad), against the sum-of-terms
engine's eval_and_grad and against finite differences.

For a controlled rotation the gradient is in the parameter as passed (θ/2),
as in the sum-of-terms engine.
"""
import random

import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator
from test_tape_eval import (apply, pauli_op, random_cr_case, tapes, with_params)


def tape_value_and_grad(program, theta):
    out = np.empty(len(theta))
    value = program.eval_and_grad(np.sin(theta), np.cos(theta), out)
    return value, out


def numeric_grad(program, theta, eps=1e-6):
    def f(t):
        return program.eval(np.sin(t), np.cos(t))
    return np.array([(f(theta + eps * e) - f(theta - eps * e)) / (2 * eps) for e in np.eye(len(theta))])


def old_engine_value_and_grad(gates, observable, theta, k1=None, dead=False, xy=False):
    def ansatz(params):
        apply(gates, params)
        return [qml.expval(qml.dot([c for _, c in observable], [pauli_op(w) for w, _ in observable]))]
    prop = Propagator(ansatz, k1=k1)
    prop.propagate(n_threads=1, use_dead_qubit_pruner=dead, use_xy_weight_pruner=xy)
    values, grads = prop.eval_and_grad(theta)
    return float(np.ravel(values)[0]), np.asarray(grads)[0]


def every_gate_case(seed):
    """A random circuit with every kind of gate: trainable and fixed rotations
    and controlled rotations, T, and the Clifford gates."""
    n, gates, observable, _ = random_cr_case(seed, fixed=True)
    rng = random.Random(4000 + seed)
    out = []
    for gate in gates:
        if rng.random() < 0.2:
            out.append(("T", rng.randrange(n)))
        out.append(gate)
    _, n_params = with_params(out)
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params)
    return n, out, observable, theta


@pytest.mark.parametrize("seed", range(30))
def test_gradient_matches_old_engine_and_finite_differences(seed):
    n, gates, observable, theta = every_gate_case(seed)
    if len(theta) == 0:
        pytest.skip("no trainable parameter")
    (program,) = tapes(n, gates, [observable])
    value, grad = tape_value_and_grad(program, theta)
    assert value == pytest.approx(program.eval(np.sin(theta), np.cos(theta)), abs=1e-14)
    old_value, old_grad = old_engine_value_and_grad(gates, observable, theta)
    assert value == pytest.approx(old_value, abs=1e-12)
    np.testing.assert_allclose(grad, old_grad, atol=1e-12)
    np.testing.assert_allclose(grad, numeric_grad(program, theta), atol=1e-7)


@pytest.mark.parametrize("seed", range(30))
def test_gradient_with_k1_and_pruners_matches_old_engine(seed):
    n, gates, observable, theta = every_gate_case(seed)
    if len(theta) == 0:
        pytest.skip("no trainable parameter")
    rng = random.Random(seed)
    k1, dead, xy = rng.randint(1, n), rng.random() < 0.5, rng.random() < 0.5
    (program,) = tapes(n, gates, [observable], k1=k1, dead=dead, xy=xy)
    value, grad = tape_value_and_grad(program, theta)
    old_value, old_grad = old_engine_value_and_grad(gates, observable, theta, k1=k1, dead=dead, xy=xy)
    assert value == pytest.approx(old_value, abs=1e-12)
    np.testing.assert_allclose(grad, old_grad, atol=1e-12)
    np.testing.assert_allclose(grad, numeric_grad(program, theta), atol=1e-7)


def test_gradient_on_brickwork():
    n, L = 8, 4
    gates = []
    for layer in range(L):
        for q in range(n):
            gates += [("RX", q), ("RY", q)]
        gates += [("CNOT", (q, q + 1)) for q in range(layer % 2, n - 1, 2)]
    word = "I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1)
    theta = np.random.default_rng(6).uniform(-np.pi, np.pi, 2 * n * L)
    (program,) = tapes(n, gates, [[(word, 1.0)]], k1=4, dead=True, xy=True)
    value, grad = tape_value_and_grad(program, theta)
    old_value, old_grad = old_engine_value_and_grad(gates, [(word, 1.0)], theta, k1=4, dead=True, xy=True)
    assert value == pytest.approx(old_value, abs=1e-12)
    np.testing.assert_allclose(grad, old_grad, atol=1e-12)


def test_unused_parameters_get_zero_gradient():
    # θ1's gate touches nothing the observable reaches: its gradient is 0, and
    # `out` may be longer than the parameters the tape uses.
    gates = [("RX", 0), ("RX", 1)]
    (program,) = tapes(2, gates, [[("ZI", 1.0)]])
    theta = np.array([0.3, 1.1, 0.0])
    out = np.full(3, 7.0)
    value = program.eval_and_grad(np.sin(theta), np.cos(theta), out)
    assert value == pytest.approx(np.cos(0.3))
    np.testing.assert_allclose(out, [-np.sin(0.3), 0.0, 0.0], atol=1e-15)


def test_out_must_match():
    (program,) = tapes(1, [("RX", 0)], [[("Z", 1.0)]])
    with pytest.raises(ValueError, match="as long as sins"):
        program.eval_and_grad(np.zeros(1), np.ones(1), np.empty(2))
