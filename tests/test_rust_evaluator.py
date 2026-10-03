"""Cross-check the optional fused Rust evaluator against the NumPy reference."""

import numpy as np
import pytest

from pprop.propagator import evaluator


@pytest.mark.skipif(evaluator.Evaluator is None, reason="pprop_rs.Evaluator not built")
def test_rust_and_numpy_evaluators_agree():
    # Repeated indices exercise powers, while an empty pair exercises the
    # constant-term sentinel. The final two angles hit both singular paths.
    expr = [
        (0.7, [0], [1]),
        (-0.2, [1, 1], []),
        (0.4, [], [0, 2]),
        (-0.1, [], []),
    ]
    rust_eval, rust_grad = evaluator.make_sparse_evaluator(expr, 3)

    saved = evaluator.Evaluator
    evaluator.Evaluator = None
    try:
        numpy_eval, numpy_grad = evaluator.make_sparse_evaluator(expr, 3)
    finally:
        evaluator.Evaluator = saved

    rng = np.random.default_rng(0)
    thetas = [rng.uniform(-np.pi, np.pi, 3) for _ in range(4)]
    thetas += [np.zeros(3), np.full(3, np.pi / 2)]

    for theta in thetas:
        sins, coss = np.sin(theta), np.cos(theta)
        assert np.isclose(rust_eval(sins, coss), numpy_eval(sins, coss),
                          rtol=1e-12, atol=1e-14)
        value_rust, grad_rust = rust_grad(sins, coss)
        value_numpy, grad_numpy = numpy_grad(sins, coss)
        assert np.isclose(value_rust, value_numpy, rtol=1e-12, atol=1e-14)
        assert np.allclose(grad_rust, grad_numpy, rtol=1e-10, atol=1e-13)


@pytest.mark.skipif(evaluator.Evaluator is None, reason="pprop_rs.Evaluator not built")
@pytest.mark.parametrize("coeffs,idx,cnt,error", [
    ([1.0], [0], [], "equal length"),
    ([1.0], [0], [2], "add up"),
    ([1.0], [3], [1], "out of range"),
])
def test_native_constructor_rejects_invalid_layouts(coeffs, idx, cnt, error):
    with pytest.raises(ValueError, match=error):
        evaluator.Evaluator(np.asarray(coeffs, dtype=np.float64),
                            np.asarray(idx, dtype=np.uint32),
                            np.asarray(cnt, dtype=np.uint32), 1)


@pytest.mark.skipif(evaluator.Evaluator is None, reason="pprop_rs.Evaluator not built")
@pytest.mark.parametrize("coeffs,idx,cnt,expected", [
    ([], [], [], 0.0),
    ([2.5], [0], [1], 2.5),
])
def test_native_zero_parameter_expressions(coeffs, idx, cnt, expected):
    kernel = evaluator.Evaluator(np.asarray(coeffs, dtype=np.float64),
                                 np.asarray(idx, dtype=np.uint32),
                                 np.asarray(cnt, dtype=np.uint32), 0)
    empty = np.empty(0, dtype=np.float64)
    assert kernel.eval(empty, empty) == expected
    assert kernel.eval_and_grad(empty, empty, empty) == expected


@pytest.mark.skipif(evaluator.Evaluator is None, reason="pprop_rs.Evaluator not built")
def test_native_rejects_incorrect_angle_and_gradient_lengths():
    kernel = evaluator.Evaluator(np.array([1.0]), np.array([0], dtype=np.uint32),
                                 np.array([1], dtype=np.uint32), 1)
    one = np.array([0.3])
    empty = np.empty(0, dtype=np.float64)
    with pytest.raises(ValueError):
        kernel.eval(empty, one)
    with pytest.raises(ValueError):
        kernel.eval(one, empty)
    with pytest.raises(ValueError):
        kernel.eval_and_grad(one, one, empty)
    readonly = np.empty(1)
    readonly.flags.writeable = False
    with pytest.raises(ValueError):
        kernel.eval_and_grad(one, one, readonly)


@pytest.mark.skipif(evaluator.Evaluator is None, reason="pprop_rs.Evaluator not built")
def test_native_returned_gradients_are_independently_owned():
    _, gradient = evaluator.make_sparse_evaluator([(1.0, [0], [])], 1)
    first_theta, second_theta = np.array([0.3]), np.array([1.2])
    _, first = gradient(np.sin(first_theta), np.cos(first_theta))
    _, second = gradient(np.sin(second_theta), np.cos(second_theta))
    np.testing.assert_allclose(first, np.cos(first_theta))
    np.testing.assert_allclose(second, np.cos(second_theta))
    assert not np.shares_memory(first, second)
