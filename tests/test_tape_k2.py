"""The k2 truncation in the tape (every box holds k2 + 1 numbers, one per
order), against the sum-of-terms engine: Propagator uses the tape, and its
exprs always come from the sum-of-terms engine with the same settings."""
import random

import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator
from pprop.propagator.evaluator import make_sparse_evaluator
from test_tape_eval import apply, pauli_op
from test_tape_grad import every_gate_case


def both_engines(gates, observable, theta, **kw):
    """(tape value, tape gradient, old value, old gradient) for the first observable."""
    prop_kw = {k: kw.pop(k) for k in ("k1", "k2", "k2_mode") if k in kw}

    def ansatz(params):
        apply(gates, params)
        return [qml.expval(qml.dot([c for _, c in observable], [pauli_op(w) for w, _ in observable]))]
    p = Propagator(ansatz, **prop_kw)
    p.propagate(n_threads=1, **kw)
    assert p.engine == "tape"
    values, grads = p.eval_and_grad(theta)
    _, g = make_sparse_evaluator(p.exprs[0], p.num_params)
    old_value, old_grad = g(np.sin(theta), np.cos(theta))
    return float(values[0]), np.asarray(grads[0]), float(old_value), np.asarray(old_grad)


@pytest.mark.parametrize("mode", ["frequency", "sine"])
@pytest.mark.parametrize("seed", range(25))
def test_k2_matches_old_engine(seed, mode):
    n, gates, observable, theta = every_gate_case(seed)
    if len(theta) == 0:
        pytest.skip("no trainable parameter")
    k2 = random.Random(seed).choice([0, 1, 2, 3, 5])
    v, g, ov, og = both_engines(gates, observable, theta, k2=k2, k2_mode=mode)
    assert v == pytest.approx(ov, abs=1e-12)
    np.testing.assert_allclose(g, og, atol=1e-12)


@pytest.mark.parametrize("mode", ["frequency", "sine"])
@pytest.mark.parametrize("seed", range(15))
def test_k2_with_k1_and_pruners_matches_old_engine(seed, mode):
    n, gates, observable, theta = every_gate_case(seed)
    if len(theta) == 0:
        pytest.skip("no trainable parameter")
    rng = random.Random(100 + seed)
    kw = dict(k1=rng.randint(1, n), k2=rng.choice([1, 2, 4]), k2_mode=mode,
              use_dead_qubit_pruner=rng.random() < 0.5, use_xy_weight_pruner=rng.random() < 0.5)
    v, g, ov, og = both_engines(gates, observable, theta, **kw)
    assert v == pytest.approx(ov, abs=1e-12)
    np.testing.assert_allclose(g, og, atol=1e-12)


@pytest.mark.parametrize("mode", ["frequency", "sine"])
def test_k2_on_brickwork(mode):
    n, L = 6, 4
    gates = []
    for layer in range(L):
        for q in range(n):
            gates += [("RX", q), ("RY", q)]
        gates += [("CNOT", (q, q + 1)) for q in range(layer % 2, n - 1, 2)]
    word = "I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1)
    theta = np.random.default_rng(8).uniform(-np.pi, np.pi, 2 * n * L)
    for k2 in (2, 4, 6):
        v, g, ov, og = both_engines(gates, [(word, 1.0)], theta, k2=k2, k2_mode=mode, k1=3)
        assert v == pytest.approx(ov, abs=1e-12)
        np.testing.assert_allclose(g, og, atol=1e-12)


def test_k2_zero_keeps_only_paths_without_factors():
    # RX(θ0): <Z> = cos θ0, which has one factor; k2 = 0 drops it entirely.
    gates = [("RX", 0)]
    v, _, ov, _ = both_engines(gates, [("Z", 1.0)], np.array([0.4]), k2=0)
    assert v == 0.0 and ov == 0.0
    # In sine mode a cos factor is free: k2 = 0 keeps cos θ0.
    v, _, ov, _ = both_engines(gates, [("Z", 1.0)], np.array([0.4]), k2=0, k2_mode="sine")
    assert v == pytest.approx(np.cos(0.4)) and ov == pytest.approx(np.cos(0.4))
