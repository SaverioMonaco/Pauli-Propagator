"""coeff_threshold in the tape: a word is dropped once an upper bound on its
coefficient, over all parameters, falls below the threshold. The dropped
words' bounds add up to dropped_bound, and without k1 and k2 the expectation
value is off by at most that, for any parameters: checked here against exact
statevectors."""
import random

import numpy as np
import pytest

from test_tape_eval import statevector, tapes, with_params
from test_tape_grad import every_gate_case

import pprop_rs
from pprop.propagator import _GATE_KIND


def threshold_tape(n, gates, observable, threshold, **kw):
    gates_p, _ = with_params(gates)
    wires = [w if isinstance(w, tuple) else (w, -1) for _, w, _, _ in gates_p]
    from test_tape_eval import obs_spec
    (program,) = pprop_rs.compile_tapes(
        n,
        [_GATE_KIND[g] for g, _, _, _ in gates_p],
        [w0 for w0, _ in wires],
        [w1 for _, w1 in wires],
        [p for _, _, p, _ in gates_p],
        [f for _, _, _, f in gates_p],
        [obs_spec(observable)],
        coeff_threshold=threshold,
        **kw,
    )
    return program


def value(program, theta):
    return program.eval(np.sin(theta), np.cos(theta))


@pytest.mark.parametrize("threshold", [0.01, 0.05, 0.2])
@pytest.mark.parametrize("seed", range(25))
def test_error_is_at_most_dropped_bound(seed, threshold):
    n, gates, observable, theta = every_gate_case(seed)
    program = threshold_tape(n, gates, observable, threshold)
    rng = np.random.default_rng(seed)
    for _ in range(4):                       # the bound holds for any parameters
        th = rng.uniform(-np.pi, np.pi, len(theta))
        exact = statevector(n, gates, observable, th)
        assert abs(value(program, th) - exact) <= program.dropped_bound + 1e-12


@pytest.mark.parametrize("threshold", [0.02, 0.1])
def test_error_bound_with_many_constant_factors(threshold):
    # About half the rotations fixed: many words are cut, and the error must
    # still stay within dropped_bound.
    from test_tape_eval import random_fixed_case
    dropped = 0
    for seed in range(30):
        n, gates, observable, theta = random_fixed_case(seed)
        program = threshold_tape(n, gates, observable, threshold)
        dropped += program.n_dropped > 0
        exact = statevector(n, gates, observable, theta)
        assert abs(value(program, theta) - exact) <= program.dropped_bound + 1e-12, seed
    assert dropped >= 10                     # the bound is really being tested


def debug(n, gates, observable, threshold, **kw):
    """tape_debug with a threshold; gates as in compile tests: (name, wires, param, fixed)."""
    from test_tape_eval import obs_spec
    wires = [w if isinstance(w, tuple) else (w, -1) for _, w, _, _ in gates]
    return pprop_rs.tape_debug(
        n, [_GATE_KIND[g] for g, _, _, _ in gates], [w0 for w0, _ in wires], [w1 for _, w1 in wires],
        [p for _, _, p, _ in gates], [f for _, _, _, f in gates], obs_spec(observable),
        coeff_threshold=threshold, **kw)


def test_bounds_by_hand():
    # <Z + 0.5 Y> through a fixed RX(φ): new Z = c·Z − s·Y and new Y = s·Z + c·Y,
    # so the bounds are |c|·1 + |s|·0.5 and |s|·1 + |c|·0.5 (sums, not a max).
    phi = 0.7
    c, s = abs(np.cos(phi)), abs(np.sin(phi))
    d = debug(1, [("RX", 0, -1, phi)], [("Z", 1.0), ("Y", 0.5)], 0.0)
    assert d["bounds"] == pytest.approx([c + 0.5 * s, s + 0.5 * c])
    # Trainable: both words can reach √(1² + 0.5²) for some θ.
    d = debug(1, [("RX", 0, 0, None)], [("Z", 1.0), ("Y", 0.5)], 0.0)
    assert d["bounds"] == pytest.approx([np.sqrt(1.25)] * 2)
    # The observable's norm caps every bound without k2: a second trainable
    # rotation cannot push them above √1.25.
    d = debug(1, [("RX", 0, 0, None), ("RX", 0, 1, None)], [("Z", 1.0), ("Y", 0.5)], 0.0)
    assert max(d["bounds"]) == pytest.approx(np.sqrt(1.25))


def test_small_partner_is_not_made_and_counted():
    # RX(0.01) fixed on <Z>: Y would get at most |sin 0.01| < 0.05, so it is
    # not made; Z keeps its cos part (a single), and 0.01 goes to dropped_bound.
    d = debug(1, [("RX", 0, -1, 0.01)], [("Z", 1.0)], 0.05)
    assert d["n_boxes"] == 1 and d["singles"] == [(0, 0)]
    assert d["n_dropped"] == 1 and d["dropped_bound"] == pytest.approx(abs(np.sin(0.01)))
    assert d["bounds"] == pytest.approx([abs(np.cos(0.01))])


def test_threshold_zero_drops_nothing():
    n, gates, observable, theta = every_gate_case(5)
    program = threshold_tape(n, gates, observable, 0.0)
    (full,) = tapes(n, gates, [observable])
    assert program.n_dropped == 0 and program.dropped_bound == 0.0
    assert program.n_pairs == full.n_pairs
    assert value(program, theta) == pytest.approx(value(full, theta), abs=1e-14)


def test_a_large_threshold_drops_and_shrinks_the_tape():
    # With trainable angles a word can reach the full norm for some θ, so a
    # bound over all θ never gets small: only constant factors (fixed angles,
    # T, controlled rotations, small observable coefficients) let the
    # threshold drop anything. Here every RY gets a fixed angle.
    n, L = 6, 4
    angles = np.random.default_rng(3).uniform(-np.pi, np.pi, (L, n))
    gates = []
    for layer in range(L):
        for q in range(n):
            gates += [("RX", q), ("RY", q, angles[layer, q])]
        gates += [("CNOT", (q, q + 1)) for q in range(layer % 2, n - 1, 2)]
    word = "I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1)
    (full,) = tapes(n, gates, [[(word, 1.0)]])
    small = threshold_tape(n, gates, [(word, 1.0)], 0.3)
    assert small.n_dropped > 0 and small.n_pairs < full.n_pairs
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, n * L)
    assert abs(value(small, theta) - statevector(n, gates, [(word, 1.0)], theta)) <= small.dropped_bound + 1e-12


def test_trainable_angles_alone_drop_nothing():
    # Every coefficient of a circuit with only trainable rotations can reach
    # the full norm for some θ: the threshold cannot drop anything (the
    # sum-of-terms engine compares the θ-free part of each term, ±1 here, and
    # drops nothing either).
    gates = [("RX", 0), ("RY", 1), ("CNOT", (0, 1)), ("RX", 1), ("RY", 0)]
    program = threshold_tape(2, gates, [("ZZ", 1.0)], 0.9)
    assert program.n_dropped == 0


@pytest.mark.parametrize("seed", range(10))
def test_gradient_with_threshold_matches_finite_differences(seed):
    n, gates, observable, theta = every_gate_case(seed)
    if len(theta) == 0:
        pytest.skip("no trainable parameter")
    rng = random.Random(seed)
    kw = dict(k1=rng.choice([-1, n]), k2=rng.choice([-1, 3]))
    program = threshold_tape(n, gates, observable, 0.05, **kw)
    out = np.empty(len(theta))
    program.eval_and_grad(np.sin(theta), np.cos(theta), out)
    eps = 1e-6
    fd = [(value(program, theta + eps * e) - value(program, theta - eps * e)) / (2 * eps) for e in np.eye(len(theta))]
    np.testing.assert_allclose(out, fd, atol=1e-7)
