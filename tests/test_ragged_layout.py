"""Cross-check the Rust ``ragged_layout`` fast path against the Python loop."""

import numpy as np
import pytest

from pprop.propagator import evaluator


@pytest.mark.skipif(evaluator._rust_ragged_layout is None,
                    reason="pprop_rs.ragged_layout not built")
def test_rust_ragged_layout_matches_python():
    # Powers, the constant-term sentinel, cos-only and sin-only terms, and an
    # empty expression; then enough random ones to cover the index offsets.
    exprs = [
        ([(0.7, [0], [1]), (-0.2, [1, 1], []), (0.4, [], [0, 2]), (-0.1, [], [])], 3),
        ([], 4),
        ([], 0),
        ([(2.5, [], [])], 0),
    ]
    rng = np.random.default_rng(1)
    for _ in range(50):
        num_params = int(rng.integers(1, 8))
        exprs.append(([
            (float(rng.normal()),
             [int(j) for j in rng.integers(0, num_params, rng.integers(0, 4))],
             [int(j) for j in rng.integers(0, num_params, rng.integers(0, 4))])
            for _ in range(int(rng.integers(0, 10)))
        ], num_params))

    for expr, num_params in exprs:
        fast = evaluator.build_ragged_arrays(expr, num_params)
        slow = evaluator._build_ragged_arrays_py(expr, num_params)
        for a, b in zip(fast, slow):
            assert a.dtype == b.dtype
            np.testing.assert_array_equal(a, b)
