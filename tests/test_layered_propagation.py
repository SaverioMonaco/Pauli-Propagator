"""
Checks the propagation kernel against the plain definition it implements:
evolve every Pauli word through one gate at a time, in reverse circuit order,
truncate after every gate, and keep the words with no X/Y at the end.

The kernel does that work in a different order - wire-disjoint gates as one
layer, the exact pruners only when a gate is about to act on a word - which
must not change the expression. The reference below builds the one-gate-at-a-
time version from ``pprop_rs.evolve_single_gate_debug`` (itself checked
against the Python rule tables in ``test_rule_tables.py``).
"""
import random
from collections import Counter, defaultdict

import numpy as np
import pennylane as qml
import pprop_rs
import pytest

from pprop import Propagator
from pprop.propagator import _GATE_KIND, _int_to_words, _words_needed

_ONE_QUBIT = [qml.H, qml.S, qml.T, qml.SX]
_ROTATIONS = [qml.RX, qml.RY, qml.RZ]
_TWO_QUBIT = [qml.CNOT, qml.CY, qml.CZ, qml.SWAP]
_CONTROLLED_ROTATIONS = [qml.CRX, qml.CRY, qml.CRZ]
_FIXED_ANGLES = [np.pi, np.pi / 2, np.pi / 3, 0.3, -np.pi]


def _random_circuit(rng, n, depth):
    """Every gate kind, with some rotations given fixed angles, and three
    multi-term observables of weight up to 4."""
    ops = []
    for _ in range(depth):
        for q in range(n):
            gate = rng.choice(_ONE_QUBIT + _ROTATIONS + _ROTATIONS)
            fixed = rng.choice(_FIXED_ANGLES) if gate in _ROTATIONS and rng.random() < 0.25 else None
            ops.append((gate, [q], fixed))
        for _ in range(max(1, n // 2)):
            gate = rng.choice(_TWO_QUBIT + _CONTROLLED_ROTATIONS)
            fixed = (rng.choice(_FIXED_ANGLES)
                     if gate in _CONTROLLED_ROTATIONS and rng.random() < 0.3 else None)
            ops.append((gate, rng.sample(range(n), 2), fixed))

    paulis = [qml.PauliX, qml.PauliY, qml.PauliZ]
    observables = []
    for _ in range(3):
        terms = []
        for _ in range(rng.randint(1, 4)):
            wires = rng.sample(range(n), rng.randint(1, min(4, n)))
            op = rng.choice(paulis)(wires[0])
            for w in wires[1:]:
                op = op @ rng.choice(paulis)(w)
            terms.append(rng.uniform(-2, 2) * op)
        observables.append(sum(terms[1:], terms[0]))

    def circuit(params):
        k = 0
        for gate, wires, fixed in ops:
            if gate in _ROTATIONS or gate in _CONTROLLED_ROTATIONS:
                if fixed is None:
                    gate(params[k], wires=wires)
                    k += 1
                else:
                    gate(fixed, wires=wires)
            else:
                gate(wires=wires)
        return [qml.expval(o) for o in observables]

    return circuit


def _gate_specs(prop):
    """The (kind, wire0, wire1, param, fixed) list propagate() hands the kernel."""
    specs = []
    for g in prop.gates:
        wire1 = int(g.wires[1]) if len(g.wires) > 1 else -1
        if g.parameter is None:
            param, fixed = -1, None
        elif isinstance(g.parameter, (int, np.integer)):
            param, fixed = int(g.parameter), None
        else:
            param, fixed = -1, float(g.parameter)
        specs.append((_GATE_KIND[g.qml_gate.name], int(g.wires[0]), wire1, param, fixed))
    return specs


def _one_gate_at_a_time(prop, coeff_threshold):
    """Reference expressions, evolving the whole map one gate at a time."""
    n_words = _words_needed(prop.num_qubits)
    k1 = prop.k1 if prop.k1 is not None else -1
    k2 = prop.k2 if prop.k2 is not None else -1
    specs = _gate_specs(prop)

    def weight(key):
        x, z = key
        return sum(bin(a | b).count("1") for a, b in zip(x, z))

    def wires_of(key):
        x, z = key
        support = 0
        for i, (a, b) in enumerate(zip(x, z)):
            support |= (a | b) << (64 * i)
        return support

    exprs = []
    for paulidict in prop.paulidicts:
        words = defaultdict(list)
        for op, terms in paulidict.items():
            key = (tuple(_int_to_words(op.x, n_words)), tuple(_int_to_words(op.z, n_words)))
            words[key] += [(float(c), list(s), list(cc)) for c, s, cc in terms]

        for kind, wire0, wire1, param, fixed in reversed(specs):
            # A gate touching no word leaves the map as it is, truncation
            # included.
            gate_wires = (1 << wire0) | ((1 << wire1) if wire1 >= 0 else 0)
            if not any(wires_of(key) & gate_wires for key in words):
                continue
            evolved = defaultdict(list)
            for (x, z), terms in words.items():
                rows = pprop_rs.evolve_single_gate_debug(
                    prop.num_qubits, kind, wire0, wire1, param, list(x), list(z), terms, fixed)
                for ox, oz, c, s, cc in rows:
                    evolved[(tuple(ox), tuple(oz))].append((c, s, cc))
            words = defaultdict(list)
            for key, terms in evolved.items():
                if k1 >= 0 and weight(key) > k1:
                    continue
                terms = [t for t in terms if k2 < 0 or len(t[1]) + len(t[2]) <= k2]
                if coeff_threshold is not None:
                    terms = [t for t in terms if abs(t[0]) >= coeff_threshold]
                if terms:
                    words[key] = terms

        exprs.append([t for (x, _z), terms in words.items() if not any(x) for t in terms])
    return exprs


def _as_multiset(expr):
    return Counter((float(c), tuple(s), tuple(cc)) for c, s, cc in expr)


@pytest.mark.parametrize("seed", range(40))
def test_kernel_matches_one_gate_at_a_time(seed):
    rng = random.Random(seed)
    n = rng.choice([3, 4, 5])
    k1 = rng.choice([None, 1, 2, 3, 4])
    k2 = rng.choice([None, 1, 2, 3, 5])
    coeff_threshold = rng.choice([None, None, 1e-3, 0.05, 0.3])
    circuit = _random_circuit(rng, n, rng.choice([1, 2, 3]))

    prop = Propagator(circuit, k1=k1, k2=k2)
    expected = _one_gate_at_a_time(prop, coeff_threshold)

    # The exact pruners only remove words that can never reach the
    # expectation, so every combination has to give the same expression.
    for dead, xy in [(False, False), (True, False), (False, True), (True, True)]:
        prop = Propagator(circuit, k1=k1, k2=k2)
        prop.propagate(use_dead_qubit_pruner=dead, use_xy_weight_pruner=xy,
                       coeff_threshold=coeff_threshold)
        assert len(prop.exprs) == len(expected)
        for got, want in zip(prop.exprs, expected):
            assert _as_multiset(got) == _as_multiset(want), (dead, xy)


def test_observable_words_are_truncated_after_the_first_gate_step():
    # In reverse order RX(0) and CNOT(1, 2) form one layer. Z0 meets RX, the
    # first gate step; X1 X2 only meets CNOT, the second. With k1=1 the
    # one-gate-at-a-time loop drops X1 X2 (weight 2) right after RX, before
    # CNOT could turn it into X1 and H into Z1, which would add a constant.
    def circuit(params):
        qml.H(wires=1)
        qml.CNOT(wires=[1, 2])
        qml.RX(params[0], wires=0)
        return [qml.expval(qml.PauliZ(0) + qml.PauliX(1) @ qml.PauliX(2))]

    prop = Propagator(circuit, k1=1)
    expected = _one_gate_at_a_time(prop, None)
    prop.propagate()
    assert _as_multiset(prop.exprs[0]) == _as_multiset(expected[0])
    assert _as_multiset(prop.exprs[0]) == Counter({(1.0, (), (0,)): 1})
