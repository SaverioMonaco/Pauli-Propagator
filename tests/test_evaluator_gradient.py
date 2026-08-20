"""
Regression test for the evaluator's derivative at exact zeros of sin/cos.
"""
import numpy as np

from pprop.propagator.evaluator import make_sparse_evaluator
from legacy_sparse_arrays import build_sparse_arrays, eval_grad_sparse_arrays, eval_sparse_arrays


def energy(theta):
    """sin(t0)cos(t1) + 2 sin(t1)^2 cos(t0) - powers 1 and 2, for sin and cos."""
    return (np.sin(theta[0]) * np.cos(theta[1])
            + 2.0 * np.sin(theta[1]) ** 2 * np.cos(theta[0]))


def test_power1_gradient_at_exact_zero():
    """
    d/dtheta sin(theta) is cos(theta), which is 1 at theta = 0, not 0. The
    derivative used to guard on the base (``sin_g != 0``) to stop padding
    entries from evaluating ``x ** -1``, which also caught genuine power-1
    factors sitting exactly on a zero of sin and returned no gradient for them.

    Exact zeros are reachable in practice - initialising every angle at 0 is a
    common choice. Random initialisation misses this, since sin(pi) evaluates
    to 1.22e-16 rather than 0.0.
    """
    _, eval_grad = make_sparse_evaluator([(1.0, [0], [1]), (2.0, [1, 1], [0])], 2)

    for values in ([0.0, 0.0], [0.0, np.pi / 2], [np.pi / 2, 0.0], [0.4, 1.3]):
        theta = np.array(values)
        value, grad = eval_grad(np.sin(theta), np.cos(theta))
        assert np.isclose(value, energy(theta), atol=1e-12)

        for i in range(2):
            step = np.zeros(2)
            step[i] = 1e-6
            central = (energy(theta + step) - energy(theta - step)) / 2e-6
            assert np.isclose(grad[i], central, atol=1e-8), (
                f"theta={values}, d/dtheta_{i}: got {grad[i]}, "
                f"central differences give {central}"
            )


def test_ragged_evaluator_agrees_with_legacy_padded_arrays():
    """
    The ragged evaluator (pprop.propagator.evaluator) replaced the padded,
    per-term-width layout that build_sparse_arrays used to feed. That layout
    isn't part of the package anymore, but it's simple enough to trust on its
    own, so it's kept here purely as an independent check that the ragged
    rewrite didn't change any values - uneven term widths and repeated
    (power > 1) indices included.
    """
    num_params = 4
    expr = [
        (3.0, [0, 1], [2]),        # width 3: sin(a) sin(b) cos(c)
        (2.0, [0], []),            # width 1: sin(a)
        (5.0, [], []),             # width 0: constant
        (1.5, [1, 1, 1], [3]),     # width 4: sin(b)^3 cos(d)
    ]

    eval_new, evalgrad_new = make_sparse_evaluator(expr, num_params)
    coeffs, idx_sin, pow_sin, idx_cos, pow_cos = build_sparse_arrays(expr, num_params)

    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.uniform(-np.pi, np.pi, num_params)
        sins, coss = np.sin(theta), np.cos(theta)

        v_new = eval_new(sins, coss)
        v_legacy = eval_sparse_arrays(coeffs, idx_sin, pow_sin, idx_cos, pow_cos, sins, coss)
        assert np.isclose(v_new, v_legacy, atol=1e-12), (
            f"theta={theta}: ragged={v_new}, legacy padded={v_legacy}"
        )

        v_new, g_new = evalgrad_new(sins, coss)
        v_legacy, g_legacy = eval_grad_sparse_arrays(
            coeffs, idx_sin, pow_sin, idx_cos, pow_cos, sins, coss, num_params
        )
        assert np.isclose(v_new, v_legacy, atol=1e-12)
        assert np.allclose(g_new, g_legacy, atol=1e-12), (
            f"theta={theta}: ragged grad={g_new}, legacy padded grad={g_legacy}"
        )
