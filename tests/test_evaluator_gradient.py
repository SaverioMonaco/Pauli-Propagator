"""
Regression test for the evaluator's derivative at exact zeros of sin/cos.
"""
import numpy as np

from pprop.propagator.evaluator import make_sparse_evaluator


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
