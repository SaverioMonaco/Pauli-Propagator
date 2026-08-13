# %%
"""
Covers pprop.optimization: the Adam optimiser and the L-BFGS-B one added
alongside it. Circuits here are deliberately tiny (2-3 qubits, 1-2 layers,
weight-truncated) since propagation cost grows with the number of
non-Clifford rotations and these tests only need a well-defined minimum, not
a hard one.
"""
import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator
from pprop.optimization import adam, lbfgs

# L picks the single observable out of f_vals, so dL/df is a constant unit
# vector; supplying it keeps both optimisers on their exact-gradient path.
L_first = lambda f: f[0]  # noqa: E731
GRAD_L_first = lambda f: np.array([1.0])  # noqa: E731


def _single_ry():
    """RY(θ) on |0> measured in Z, i.e. E(θ) = cos(θ): minimum -1 at θ = π."""
    def ansatz(params):
        qml.RY(params[0], wires=0)
        return qml.expval(qml.PauliZ(0))

    prop = Propagator(ansatz)
    prop.propagate()
    return prop


def _small_vqe():
    """3-qubit transverse-field Ising chain with a shallow hardware-efficient ansatz."""
    nq, layers = 3, 2
    H = qml.sum(
        *[-1.0 * (qml.PauliZ(i) @ qml.PauliZ(i + 1)) for i in range(nq - 1)],
        *[-1.0 * qml.PauliX(i) for i in range(nq)],
    )

    def ansatz(params):
        k = 0
        for q in range(nq):
            qml.Hadamard(wires=q)
        for _ in range(layers):
            for q in range(nq):
                qml.RY(params[k], wires=q)
                k += 1
            for q in range(nq - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(H)

    prop = Propagator(ansatz)
    prop.propagate(use_dead_qubit_pruner=True, use_xy_weight_pruner=True)

    exact = float(np.linalg.eigvalsh(qml.matrix(H, wire_order=range(nq)))[0])
    return prop, exact


# %%
def test_lbfgs_finds_analytic_minimum():
    """E(θ) = cos(θ) has a unique minimum of -1; L-BFGS should land on it."""
    prop = _single_ry()

    result = lbfgs(L_first, prop, np.array([2.0]), print_every=0, grad_L=GRAD_L_first)

    assert result["fun"] == pytest.approx(-1.0, abs=1e-8)
    assert result["params"][0] == pytest.approx(np.pi, abs=1e-4)
    assert result["success"]


def test_adam_still_finds_analytic_minimum():
    """Regression guard on adam after it was rewired through _value_and_grad."""
    prop = _single_ry()

    result = adam(L_first, prop, np.array([2.0]), lr=1e-1, num_steps=500,
                  print_every=0, grad_L=GRAD_L_first)

    assert result["fun"] == pytest.approx(-1.0, abs=1e-6)
    assert len(result["loss_history"]) == 500


def test_adam_finite_difference_path_matches_exact_gradient():
    """The default grad_L=None branch should agree with the exact gradient."""
    prop = _single_ry()
    kwargs = dict(lr=1e-1, num_steps=200, print_every=0)

    exact = adam(L_first, prop, np.array([2.0]), grad_L=GRAD_L_first, **kwargs)
    findiff = adam(L_first, prop, np.array([2.0]), **kwargs)

    assert findiff["fun"] == pytest.approx(exact["fun"], abs=1e-6)


# %%
def test_lbfgs_reaches_ground_state_of_small_vqe():
    """The 3-qubit TFIM ground state is reachable by this ansatz; L-BFGS should find it."""
    prop, exact = _small_vqe()
    rng = np.random.default_rng(0)

    # Non-convex landscape, so take the best of a few restarts rather than
    # asserting that any single random start converges globally.
    runs = [
        lbfgs(L_first, prop, rng.uniform(0, 2 * np.pi, prop.num_params),
              print_every=0, grad_L=GRAD_L_first)
        for _ in range(5)
    ]
    best = min(runs, key=lambda r: r["fun"])

    assert best["fun"] >= exact - 1e-8, "cannot beat the exact ground-state energy"
    assert best["fun"] == pytest.approx(exact, abs=1e-3)


def test_lbfgs_reaches_a_stationary_point_cheaply():
    """
    The reason to prefer L-BFGS here is cost: on a deterministic, exactly
    differentiable loss it reaches a stationary point in a small fraction of
    the evaluations Adam needs.

    Deliberately *not* asserted: that L-BFGS beats Adam's loss from the same
    start. It drives into the nearest local minimum, which on this non-convex
    landscape is sometimes worse than where a damped Adam run drifts. Quality
    is a property of restarts, not of a single run — see
    ``test_lbfgs_reaches_ground_state_of_small_vqe``.
    """
    prop, _ = _small_vqe()
    x0 = np.random.default_rng(1).uniform(0, 2 * np.pi, prop.num_params)

    steps = 1000
    a = adam(L_first, prop, x0, lr=1e-1, num_steps=steps, print_every=0,
             grad_L=GRAD_L_first)
    b = lbfgs(L_first, prop, x0, print_every=0, grad_L=GRAD_L_first)

    assert b["nfev"] < steps // 4
    assert b["fun"] < b["loss_history"][0], "should improve on its starting point"

    # Converged means the gradient has actually vanished, not merely that the
    # loss stopped moving.
    _, grad = prop.eval_and_grad(np.asarray(b["params"]))
    assert np.max(np.abs(grad[0])) < 1e-4

    # Both land somewhere sensible; neither is allowed to diverge.
    assert b["fun"] < 0 and a["fun"] < 0


# %%
def test_lbfgs_history_is_aligned_and_per_evaluation():
    prop, _ = _small_vqe()
    x0 = np.random.default_rng(2).uniform(0, 2 * np.pi, prop.num_params)

    result = lbfgs(L_first, prop, x0, print_every=0, grad_L=GRAD_L_first)

    assert len(result["loss_history"]) == len(result["params_history"])
    assert len(result["loss_history"]) == result["nfev"]

    # The history spans line-search trials, so it is not monotonic and the
    # returned point need not be the last one evaluated. It must still be an
    # improvement on the start, and can never beat the best point seen.
    assert result["fun"] < result["loss_history"][0]
    assert result["fun"] >= min(result["loss_history"]) - 1e-9


def test_lbfgs_does_not_mutate_params_init():
    prop = _single_ry()
    x0 = np.array([2.0])

    lbfgs(L_first, prop, x0, print_every=0, grad_L=GRAD_L_first)

    assert x0[0] == 2.0


def test_lbfgs_warns_when_grad_L_omitted():
    """The finite-difference fallback is a line-search hazard, so it warns."""
    prop = _single_ry()

    with pytest.warns(RuntimeWarning, match="finite differences"):
        result = lbfgs(L_first, prop, np.array([2.0]), print_every=0)

    # It should still get there; the warning is about robustness, not correctness.
    assert result["fun"] == pytest.approx(-1.0, abs=1e-6)


def test_lbfgs_respects_num_steps_cap():
    prop, _ = _small_vqe()
    x0 = np.random.default_rng(3).uniform(0, 2 * np.pi, prop.num_params)

    result = lbfgs(L_first, prop, x0, num_steps=3, print_every=0, grad_L=GRAD_L_first)

    assert result["nit"] <= 3
