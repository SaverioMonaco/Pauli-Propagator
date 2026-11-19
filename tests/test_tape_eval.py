"""Evaluating tapes (pprop_rs.compile_tapes, TapeProgram.eval) against PennyLane
statevectors and against the sum-of-terms engine.

So far the tape compiles one-qubit rotations (RX, RY, RZ) with trainable
angles only.
"""
import random

import numpy as np
import pennylane as qml
import pytest

import pprop_rs
from pprop import Propagator
from pprop.propagator import _GATE_KIND

PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}


def obs_spec(observable):
    """[(word string, coeff)] -> the (x words, z words, coeff, [], []) list pprop_rs takes."""
    def bits(word, letters):
        return [sum(1 << q for q, ch in enumerate(word) if ch in letters)]
    return [(bits(w, "XY"), bits(w, "ZY"), c, [], []) for w, c in observable]


def tapes(n, gates, observables):
    """gates: [(name, qubit)] in circuit order, angle p for gate p."""
    return pprop_rs.compile_tapes(
        n,
        [_GATE_KIND[g] for g, _ in gates],
        [q for _, q in gates],
        [-1] * len(gates),
        list(range(len(gates))),
        [None] * len(gates),
        [obs_spec(o) for o in observables],
    )


def tape_value(program, theta):
    return program.eval(np.sin(theta), np.cos(theta))


def pauli_op(word):
    ops = [PAULI[ch](q) for q, ch in enumerate(word) if ch != "I"]
    return qml.prod(*ops) if len(ops) > 1 else ops[0]


def statevector(n, gates, observable, theta):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def f():
        for p, (g, q) in enumerate(gates):
            getattr(qml, g)(theta[p], wires=q)
        return [qml.expval(pauli_op(w)) for w, _ in observable]
    return float(sum(c * v for (_, c), v in zip(observable, np.atleast_1d(f()))))


def old_engine(n, gates, word, theta):
    def ansatz(params):
        for p, (g, q) in enumerate(gates):
            getattr(qml, g)(params[p], wires=q)
        return [qml.expval(pauli_op(word))]
    prop = Propagator(ansatz)
    prop.propagate(n_threads=1)
    return float(np.ravel(prop(theta))[0])


def random_case(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 4)
    gates = [(rng.choice(["RX", "RY", "RZ"]), rng.randrange(n)) for _ in range(rng.randint(1, 12))]
    words = set()
    while len(words) < rng.randint(1, 3):
        w = "".join(rng.choice("IXYZ") for _ in range(n))
        if set(w) != {"I"}:
            words.add(w)
    observable = [(w, rng.uniform(-1, 1)) for w in sorted(words)]
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, len(gates))
    return n, gates, observable, theta


def test_example_by_hand():
    # RX(θ0) q0, RY(θ1) q1, RX(θ2) q1, <Z1>. Backwards: RX(θ2) gives
    # cos θ2·IZ + sin θ2·IY, then RY(θ1) turns IZ into cos θ1·IZ − sin θ1·IX
    # and leaves IY alone. Only IZ counts in |00>: <Z1> = cos θ1 · cos θ2.
    gates = [("RX", 0), ("RY", 1), ("RX", 1)]
    theta = np.array([0.3, -1.2, 2.0])
    (program,) = tapes(2, gates, [[("IZ", 1.0)]])
    assert program.n_boxes == 3 and program.n_steps == 2 and program.n_pairs == 2
    assert tape_value(program, theta) == pytest.approx(np.cos(-1.2) * np.cos(2.0), abs=1e-15)


@pytest.mark.parametrize("seed", range(30))
def test_random_circuits_match_statevector(seed):
    n, gates, observable, theta = random_case(seed)
    (program,) = tapes(n, gates, [observable])
    assert tape_value(program, theta) == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_random_circuits_match_old_engine(seed):
    n, gates, observable, theta = random_case(seed)
    word, _ = observable[0]
    (program,) = tapes(n, gates, [[(word, 1.0)]])
    assert tape_value(program, theta) == pytest.approx(old_engine(n, gates, word, theta), abs=1e-12)


def test_several_observables_and_several_thetas():
    n, gates, _, _ = random_case(3)
    observables = [[("Z" * n, 1.0)], [("X" + "I" * (n - 1), 0.5)], [("Y" * n, -2.0), ("Z" * n, 0.25)]]
    programs = tapes(n, gates, observables)
    assert len(programs) == 3
    rng = np.random.default_rng(7)
    for _ in range(3):                       # the same tapes, new angles each time
        theta = rng.uniform(-np.pi, np.pi, len(gates))
        for program, observable in zip(programs, observables):
            assert tape_value(program, theta) == pytest.approx(
                statevector(n, gates, observable, theta), abs=1e-12)


def test_too_few_angles():
    (program,) = tapes(1, [("RX", 0), ("RY", 0)], [[("Z", 1.0)]])
    with pytest.raises(ValueError, match="at least 2"):
        program.eval(np.zeros(1), np.ones(1))
