"""Propagator always uses the tape; the sum-of-terms engine is kept for
exprs/expression()."""
import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator
from pprop.propagator.evaluator import make_sparse_evaluator


def ansatz(params):
    qml.RX(params[0], wires=0)
    qml.RX(params[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RX(params[2], wires=0)
    qml.RX(params[3], wires=1)
    return [qml.expval(qml.PauliZ(1)), qml.expval(qml.PauliX(0) @ qml.PauliZ(1))]


THETA = np.array([0.3, -1.2, 2.0, 0.7])


def test_tape_is_the_default():
    p = Propagator(ansatz)
    assert p.exprs is None and p.engine is None
    p.propagate()
    assert p.engine == "tape" and len(p.programs) == 2
    assert p._exprs is None                      # not computed by propagate()


def test_exprs_are_lazy_and_from_the_old_engine():
    p = Propagator(ansatz, k1=1)
    p.propagate(use_dead_qubit_pruner=True)
    exprs = p.exprs                              # computed now, with the same k1 and pruner
    assert p.exprs is exprs                      # and kept
    for i, expr in enumerate(exprs):
        f, _ = make_sparse_evaluator(expr, p.num_params)
        assert f(np.sin(THETA), np.cos(THETA)) == pytest.approx(p(THETA)[i], abs=1e-12)


def test_expression_works_with_the_tape():
    p = Propagator(ansatz)
    p.propagate()
    assert str(p.expression(0)) == "-1.0*sin(θ1)*sin(θ3)*cos(θ0) + 1.0*cos(θ0)*cos(θ1)*cos(θ3)"


@pytest.mark.parametrize("kw", [dict(k2=2), dict(k2=1, k2_mode="sine")])
def test_k2_uses_the_tape(kw):
    p = Propagator(ansatz, **kw)
    p.propagate()
    assert p.engine == "tape" and p.programs[0].k2 == kw["k2"]
    f, _ = make_sparse_evaluator(p.exprs[0], p.num_params)
    assert p(THETA)[0] == pytest.approx(f(np.sin(THETA), np.cos(THETA)), abs=1e-12)


def fixed_ansatz(params):
    qml.RX(params[0], wires=0)
    qml.RY(0.2, wires=1)                    # a fixed angle: constant factors the threshold can cut
    qml.CNOT(wires=[0, 1])
    qml.RY(0.4, wires=0)
    return [qml.expval(qml.PauliZ(1))]


def test_coeff_threshold_uses_the_tape():
    p = Propagator(fixed_ansatz)
    p.propagate(coeff_threshold=0.5)
    assert p.engine == "tape" and p.programs[0].n_dropped > 0
    theta = np.array([0.7])

    @qml.qnode(qml.device("default.qubit", wires=2))
    def exact():
        return fixed_ansatz(theta)
    assert abs(p(theta)[0] - float(exact()[0])) <= p.programs[0].dropped_bound + 1e-12


def test_values_and_gradients_match_the_old_engine():
    p = Propagator(ansatz)
    p.propagate()
    vals, grads = p.eval_and_grad(THETA)
    for i, expr in enumerate(p.exprs):
        _, g = make_sparse_evaluator(expr, p.num_params)
        v, gr = g(np.sin(THETA), np.cos(THETA))
        assert vals[i] == pytest.approx(v, abs=1e-12)
        np.testing.assert_allclose(grads[i], gr, atol=1e-12)
