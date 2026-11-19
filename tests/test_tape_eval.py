"""Evaluating tapes (pprop_rs.compile_tapes, TapeProgram.eval) against PennyLane
statevectors and against the sum-of-terms engine.

So far the tape compiles one-qubit rotations (RX, RY, RZ) with trainable
or fixed angles, T, the Clifford gates (H, S, SX, CNOT, CY, CZ, SWAP) and the
controlled rotations (CRX, CRY, CRZ: parameter = θ/2, as everywhere in pprop), with the k1
truncation (checked against the sum-of-terms engine: a statevector cannot
truncate) and the two exact pruners.
"""
import itertools
import random

import numpy as np
import pennylane as qml
import pytest

import pprop_rs
from pprop import Propagator
from pprop.propagator import _GATE_KIND

PAULI = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
ROTATIONS = ["RX", "RY", "RZ"]
CONTROLLED_ROTATIONS = ["CRX", "CRY", "CRZ"]
CLIFFORD_1Q = ["Hadamard", "S", "SX"]
CLIFFORD_2Q = ["CNOT", "CY", "CZ", "SWAP"]


def obs_spec(observable):
    """[(word string, coeff)] -> the (x words, z words, coeff, [], []) list pprop_rs takes."""
    def bits(word, letters):
        return [sum(1 << q for q, ch in enumerate(word) if ch in letters)]
    return [(bits(w, "XY"), bits(w, "ZY"), c, [], []) for w, c in observable]


def with_params(gates):
    """gates: [(name, wires)] or [(name, wires, fixed angle)] in circuit order,
    wires a qubit or a (control, target) pair -> [(name, wires, param, fixed)]:
    the trainable rotations numbered 0, 1, ...; param -1 for the others."""
    out, p = [], 0
    for gate in gates:
        g, w = gate[0], gate[1]
        if len(gate) == 3:
            out.append((g, w, -1, float(gate[2])))
        elif g in ROTATIONS or g in CONTROLLED_ROTATIONS:
            out.append((g, w, p, None))
            p += 1
        else:
            out.append((g, w, -1, None))
    return out, p


def tapes(n, gates, observables, k1=-1, dead=False, xy=False):
    gates, _ = with_params(gates)
    wires = [w if isinstance(w, tuple) else (w, -1) for _, w, _, _ in gates]
    return pprop_rs.compile_tapes(
        n,
        [_GATE_KIND[g] for g, _, _, _ in gates],
        [w0 for w0, _ in wires],
        [w1 for _, w1 in wires],
        [p for _, _, p, _ in gates],
        [f for _, _, _, f in gates],
        [obs_spec(o) for o in observables],
        k1,
        dead,
        xy,
    )


def tape_value(program, theta):
    return program.eval(np.sin(theta), np.cos(theta))


def pauli_op(word):
    ops = [PAULI[ch](q) for q, ch in enumerate(word) if ch != "I"]
    return qml.prod(*ops) if len(ops) > 1 else ops[0]


def apply(gates, theta, statevector=False):
    """Put the gates on the tape of the current QNode. pprop's convention for
    a trainable controlled rotation: the parameter is θ/2, so the statevector
    needs CRX(2·θ[p]) where pprop's ansatz has CRX(θ[p]). A fixed angle is
    the real angle for both."""
    gates, _ = with_params(gates)
    for g, w, p, fixed in gates:
        wires = list(w) if isinstance(w, tuple) else w
        if p >= 0:
            angle = 2 * theta[p] if statevector and g in CONTROLLED_ROTATIONS else theta[p]
            getattr(qml, g)(angle, wires=wires)
        elif fixed is not None:
            getattr(qml, g)(fixed, wires=wires)
        else:
            getattr(qml, g)(wires=wires)


def statevector(n, gates, observable, theta):
    dev = qml.device("default.qubit", wires=n)

    @qml.qnode(dev)
    def f():
        apply(gates, theta, statevector=True)
        return [qml.expval(pauli_op(w)) for w, _ in observable]
    return float(sum(c * v for (_, c), v in zip(observable, np.atleast_1d(f()))))


def old_engine(gates, observable, theta, k1=None, dead=False, xy=False):
    """observable: a word string, or [(word string, coeff)]."""
    if isinstance(observable, str):
        observable = [(observable, 1.0)]

    def ansatz(params):
        apply(gates, params)
        return [qml.expval(qml.dot([c for _, c in observable], [pauli_op(w) for w, _ in observable]))]
    prop = Propagator(ansatz, k1=k1)
    prop.propagate(n_threads=1, use_dead_qubit_pruner=dead, use_xy_weight_pruner=xy)
    return float(np.ravel(prop(theta))[0])


def random_word(rng, n):
    while True:
        w = "".join(rng.choice("IXYZ") for _ in range(n))
        if set(w) != {"I"}:
            return w


def random_case(seed, cliffords=True):
    rng = random.Random(seed)
    n = rng.randint(2 if cliffords else 1, 4)
    gates = []
    for _ in range(rng.randint(1, 16)):
        kind = rng.choice(["rot", "rot", "c1", "c2"] if cliffords else ["rot"])
        if kind == "rot":
            gates.append((rng.choice(ROTATIONS), rng.randrange(n)))
        elif kind == "c1":
            gates.append((rng.choice(CLIFFORD_1Q), rng.randrange(n)))
        else:
            gates.append((rng.choice(CLIFFORD_2Q), tuple(rng.sample(range(n), 2))))
    words = sorted({random_word(rng, n) for _ in range(rng.randint(1, 3))})
    observable = [(w, rng.uniform(-1, 1)) for w in words]
    _, n_params = with_params(gates)
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params)
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
def test_random_rotation_circuits_match_statevector(seed):
    n, gates, observable, theta = random_case(seed, cliffords=False)
    (program,) = tapes(n, gates, [observable])
    assert tape_value(program, theta) == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)


GATE_ON_2_QUBITS = [(g, q) for g in CLIFFORD_1Q for q in (0, 1)] + \
                   [(g, w) for g in CLIFFORD_2Q for w in ((0, 1), (1, 0))]
WORDS_2 = ["".join(w) for w in itertools.product("IXYZ", repeat=2) if w != ("I", "I")]


@pytest.mark.parametrize("gate", GATE_ON_2_QUBITS, ids=lambda g: f"{g[0]}{g[1]}")
def test_every_clifford_on_every_word(gate):
    # A rotation layer before and after, so every word reaches the Clifford
    # with a nonzero coefficient, and every rule entry (with its sign) matters.
    layer = [("RX", 0), ("RY", 0), ("RX", 1), ("RY", 1)]
    gates = layer + [gate] + layer
    theta = np.random.default_rng(1).uniform(-np.pi, np.pi, 8)
    for word in WORDS_2:
        (program,) = tapes(2, gates, [[(word, 1.0)]])
        assert tape_value(program, theta) == pytest.approx(
            statevector(2, gates, [(word, 1.0)], theta), abs=1e-12), word


@pytest.mark.parametrize("seed", range(40))
def test_random_mixed_circuits_match_statevector(seed):
    n, gates, observable, theta = random_case(seed)
    (program,) = tapes(n, gates, [observable])
    assert tape_value(program, theta) == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_random_mixed_circuits_match_old_engine(seed):
    n, gates, observable, theta = random_case(seed)
    word, _ = observable[0]
    (program,) = tapes(n, gates, [[(word, 1.0)]])
    assert tape_value(program, theta) == pytest.approx(old_engine(gates, word, theta), abs=1e-12)


def test_brickwork():
    # The benchmark's circuit: RX and RY on every qubit, then CNOTs on
    # alternating pairs; 6 qubits, 3 layers, Z on the middle qubit.
    n, L = 6, 3
    gates = []
    for layer in range(L):
        for q in range(n):
            gates += [("RX", q), ("RY", q)]
        gates += [("CNOT", (q, q + 1)) for q in range(layer % 2, n - 1, 2)]
    word = "I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1)
    theta = np.random.default_rng(0).uniform(-np.pi, np.pi, 2 * n * L)
    (program,) = tapes(n, gates, [[(word, 1.0)]])
    expected = statevector(n, gates, [(word, 1.0)], theta)
    assert tape_value(program, theta) == pytest.approx(expected, abs=1e-12)
    assert tape_value(program, theta) == pytest.approx(old_engine(gates, word, theta), abs=1e-12)


@pytest.mark.parametrize("seed", range(40))
def test_k1_matches_old_engine(seed):
    n, gates, observable, theta = random_case(seed)
    k1 = random.Random(seed).randint(1, n)
    (program,) = tapes(n, gates, [observable], k1=k1)
    assert tape_value(program, theta) == pytest.approx(old_engine(gates, observable, theta, k1=k1), abs=1e-12)


@pytest.mark.parametrize("k1", [1, 2, 3, 4])
def test_k1_brickwork_matches_old_engine(k1):
    n, L = 8, 3
    gates = []
    for layer in range(L):
        for q in range(n):
            gates += [("RX", q), ("RY", q)]
        gates += [("CNOT", (q, q + 1)) for q in range(layer % 2, n - 1, 2)]
    word = "I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1)
    theta = np.random.default_rng(k1).uniform(-np.pi, np.pi, 2 * n * L)
    (program,) = tapes(n, gates, [[(word, 1.0)]], k1=k1)
    (full,) = tapes(n, gates, [[(word, 1.0)]])
    if k1 <= 3:                       # in 3 layers no word gets heavier than 4
        assert program.n_boxes < full.n_boxes
    assert tape_value(program, theta) == pytest.approx(old_engine(gates, word, theta, k1=k1), abs=1e-12)


def test_k1_heavy_observable_words_match_old_engine():
    # The observable's own too-heavy words: kept through the first gate that
    # touches them, cut right after. A first CNOT can make them light again.
    # (Gates are in circuit order, so the first gate met is the last listed.)
    cases = [
        ([("RX", 1), ("RY", 0), ("CNOT", (0, 1))], [("ZZ", 1.0), ("XI", 0.5)]),
        ([("RX", 0), ("CNOT", (0, 1)), ("RY", 1)], [("ZZ", 1.0), ("ZI", -0.3)]),
        ([("RY", 2), ("CZ", (0, 1)), ("RX", 0)], [("XZI", 0.7), ("IIZ", 0.2)]),
    ]
    for gates, observable in cases:
        n = len(observable[0][0])
        _, n_params = with_params(gates)
        theta = np.random.default_rng(3).uniform(-np.pi, np.pi, n_params)
        (program,) = tapes(n, gates, [observable], k1=1)
        assert tape_value(program, theta) == pytest.approx(
            old_engine(gates, observable, theta, k1=1), abs=1e-12), (gates, observable)


PRUNERS = [(True, False), (False, True), (True, True)]


@pytest.mark.parametrize("dead,xy", PRUNERS, ids=["dead", "xy", "both"])
@pytest.mark.parametrize("seed", range(25))
def test_pruners_match_old_engine_and_change_nothing(seed, dead, xy):
    n, gates, observable, theta = random_case(seed)
    k1 = random.Random(seed).choice([-1, 1, 2, n])
    (pruned,) = tapes(n, gates, [observable], k1=k1, dead=dead, xy=xy)
    (full,) = tapes(n, gates, [observable], k1=k1)
    value = tape_value(pruned, theta)
    assert value == pytest.approx(tape_value(full, theta), abs=1e-12)          # exact pruning
    assert value == pytest.approx(
        old_engine(gates, observable, theta, k1=None if k1 < 0 else k1, dead=dead, xy=xy), abs=1e-12)
    assert pruned.n_boxes <= full.n_boxes and pruned.n_pairs <= full.n_pairs


def test_pruners_on_brickwork():
    n, L = 8, 4
    gates = []
    for layer in range(L):
        for q in range(n):
            gates += [("RX", q), ("RY", q)]
        gates += [("CNOT", (q, q + 1)) for q in range(layer % 2, n - 1, 2)]
    word = "I" * (n // 2) + "Z" + "I" * (n - n // 2 - 1)
    theta = np.random.default_rng(5).uniform(-np.pi, np.pi, 2 * n * L)
    (pruned,) = tapes(n, gates, [[(word, 1.0)]], k1=4, dead=True, xy=True)
    (full,) = tapes(n, gates, [[(word, 1.0)]], k1=4)
    assert pruned.n_pairs < full.n_pairs                                         # it prunes something
    assert tape_value(pruned, theta) == pytest.approx(tape_value(full, theta), abs=1e-12)
    assert tape_value(pruned, theta) == pytest.approx(
        old_engine(gates, word, theta, k1=4, dead=True, xy=True), abs=1e-12)


def random_fixed_case(seed):
    """A random mixed circuit in which about half of the rotations get a fixed
    angle, some of them exactly at quarter turns (where sin or cos is 0)."""
    n, gates, observable, _ = random_case(seed)
    rng = random.Random(1000 + seed)
    quarter = [0.0, np.pi / 2, np.pi, -np.pi / 2]
    out = []
    for gate in gates:
        if gate[0] in ROTATIONS and rng.random() < 0.5:
            out.append((gate[0], gate[1], rng.choice(quarter) if rng.random() < 0.3 else rng.uniform(-np.pi, np.pi)))
        else:
            out.append(gate)
    _, n_params = with_params(out)
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params)
    return n, out, observable, theta


@pytest.mark.parametrize("seed", range(30))
def test_fixed_angles_match_statevector_and_old_engine(seed):
    n, gates, observable, theta = random_fixed_case(seed)
    (program,) = tapes(n, gates, [observable])
    value = tape_value(program, theta)
    assert value == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)
    assert value == pytest.approx(old_engine(gates, observable, theta), abs=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_fixed_angles_with_k1_and_pruners_match_old_engine(seed):
    n, gates, observable, theta = random_fixed_case(seed)
    k1 = random.Random(seed).randint(1, n)
    (program,) = tapes(n, gates, [observable], k1=k1, dead=True, xy=True)
    assert tape_value(program, theta) == pytest.approx(
        old_engine(gates, observable, theta, k1=k1, dead=True, xy=True), abs=1e-12)


def test_all_fixed_circuit():
    # No trainable gate at all: the tape takes empty angle arrays.
    gates = [("RX", 0, 0.3), ("RY", 1, -1.2), ("CNOT", (0, 1)), ("RZ", 0, 2.0), ("RX", 1, np.pi / 2)]
    observable = [("ZZ", 1.0), ("XI", 0.5)]
    (program,) = tapes(2, gates, [observable])
    empty = np.zeros(0)
    assert program.eval(empty, empty) == pytest.approx(statevector(2, gates, observable, empty), abs=1e-12)


@pytest.mark.parametrize("word", ["X", "Y", "Z"])
def test_t_gate_on_each_word(word):
    gates = [("RX", 0), ("RY", 0), ("T", 0), ("RX", 0), ("RY", 0)]
    theta = np.random.default_rng(2).uniform(-np.pi, np.pi, 4)
    (program,) = tapes(1, gates, [[(word, 1.0)]])
    assert tape_value(program, theta) == pytest.approx(statevector(1, gates, [(word, 1.0)], theta), abs=1e-12)


def random_t_case(seed):
    """A random mixed circuit with T gates put in at random places."""
    n, gates, observable, theta = random_case(seed)
    rng = random.Random(2000 + seed)
    out = []
    for gate in gates:
        if rng.random() < 0.3:
            out.append(("T", rng.randrange(n)))
        out.append(gate)
    return n, out, observable, theta


@pytest.mark.parametrize("seed", range(25))
def test_t_gates_match_statevector_and_old_engine(seed):
    n, gates, observable, theta = random_t_case(seed)
    (program,) = tapes(n, gates, [observable])
    value = tape_value(program, theta)
    assert value == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)
    assert value == pytest.approx(old_engine(gates, observable, theta), abs=1e-12)
    k1 = random.Random(seed).randint(1, n)
    (program,) = tapes(n, gates, [observable], k1=k1, dead=True, xy=True)
    assert tape_value(program, theta) == pytest.approx(
        old_engine(gates, observable, theta, k1=k1, dead=True, xy=True), abs=1e-12)


CR_ON_2_QUBITS = [(g, w) for g in CONTROLLED_ROTATIONS for w in ((0, 1), (1, 0))]


@pytest.mark.parametrize("gate", CR_ON_2_QUBITS, ids=lambda g: f"{g[0]}{g[1]}")
def test_every_controlled_rotation_on_every_word(gate):
    layer = [("RX", 0), ("RY", 0), ("RX", 1), ("RY", 1)]
    gates = layer + [gate] + layer
    theta = np.random.default_rng(4).uniform(-np.pi, np.pi, 9)
    for word in WORDS_2:
        (program,) = tapes(2, gates, [[(word, 1.0)]])
        assert tape_value(program, theta) == pytest.approx(
            statevector(2, gates, [(word, 1.0)], theta), abs=1e-12), word


def random_cr_case(seed, fixed=False):
    """A random mixed circuit with controlled rotations put in at random
    places; with `fixed`, some of them get a fixed angle (some at quarter turns)."""
    n, gates, observable, _ = random_case(seed)
    rng = random.Random(3000 + seed)
    out = []
    for gate in gates:
        if rng.random() < 0.35:
            g, w = rng.choice(CONTROLLED_ROTATIONS), tuple(rng.sample(range(n), 2))
            if fixed and rng.random() < 0.5:
                angle = rng.choice([0.0, np.pi, -np.pi / 2]) if rng.random() < 0.3 else rng.uniform(-np.pi, np.pi)
                out.append((g, w, angle))
            else:
                out.append((g, w))
        out.append(gate)
    _, n_params = with_params(out)
    theta = np.random.default_rng(seed).uniform(-np.pi, np.pi, n_params)
    return n, out, observable, theta


@pytest.mark.parametrize("seed", range(25))
def test_controlled_rotations_match_statevector_and_old_engine(seed):
    n, gates, observable, theta = random_cr_case(seed)
    (program,) = tapes(n, gates, [observable])
    value = tape_value(program, theta)
    assert value == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)
    assert value == pytest.approx(old_engine(gates, observable, theta), abs=1e-12)


@pytest.mark.parametrize("seed", range(25))
def test_controlled_rotations_with_k1_and_pruners_match_old_engine(seed):
    n, gates, observable, theta = random_cr_case(seed, fixed=True)
    rng = random.Random(seed)
    k1, dead, xy = rng.randint(1, n), rng.random() < 0.5, rng.random() < 0.5
    (program,) = tapes(n, gates, [observable], k1=k1, dead=dead, xy=xy)
    assert tape_value(program, theta) == pytest.approx(
        old_engine(gates, observable, theta, k1=k1, dead=dead, xy=xy), abs=1e-12)


@pytest.mark.parametrize("seed", range(15))
def test_fixed_controlled_rotations_match_statevector(seed):
    n, gates, observable, theta = random_cr_case(seed, fixed=True)
    (program,) = tapes(n, gates, [observable])
    assert tape_value(program, theta) == pytest.approx(statevector(n, gates, observable, theta), abs=1e-12)


def test_controlled_rotation_makes_a_heavy_observable_word_light():
    # ZZ is too heavy for k1 = 1, and CRX(0 -> 1) as the first gate turns ZZ
    # partly into IZ: kept through that gate, as in the sum-of-terms engine.
    gates = [("RY", 0), ("CRX", (0, 1))]
    theta = np.array([0.4, 0.9])
    (program,) = tapes(2, gates, [[("ZZ", 1.0)]], k1=1)
    assert tape_value(program, theta) == pytest.approx(old_engine(gates, "ZZ", theta, k1=1), abs=1e-12)
    assert abs(tape_value(program, theta)) > 1e-3


def test_several_observables_and_several_thetas():
    n, gates, _, _ = random_case(3)
    observables = [[("Z" * n, 1.0)], [("X" + "I" * (n - 1), 0.5)], [("Y" * n, -2.0), ("Z" * n, 0.25)]]
    programs = tapes(n, gates, observables)
    assert len(programs) == 3
    _, n_params = with_params(gates)
    rng = np.random.default_rng(7)
    for _ in range(3):                       # the same tapes, new angles each time
        theta = rng.uniform(-np.pi, np.pi, n_params)
        for program, observable in zip(programs, observables):
            assert tape_value(program, theta) == pytest.approx(
                statevector(n, gates, observable, theta), abs=1e-12)


def test_too_few_angles():
    (program,) = tapes(1, [("RX", 0), ("RY", 0)], [[("Z", 1.0)]])
    with pytest.raises(ValueError, match="at least 2"):
        program.eval(np.zeros(1), np.ones(1))
