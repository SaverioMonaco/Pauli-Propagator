"""The tape compiler (pprop_rs.tape_debug), checked before tapes can be evaluated.

So far the tape compiles one-qubit rotations (RX, RY, RZ) with trainable
angles only.
"""
import random

import pytest

import pprop_rs
from pprop.propagator import _GATE_KIND

LETTERS = {(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}
# A pair [a, b] is written so that the gate turns a into cos·a + sin·b.
PAIR_LETTERS = {"RX": ("Z", "Y"), "RY": ("X", "Z"), "RZ": ("Y", "X")}


def word_str(x, z, n):
    """(x words, z words) -> e.g. "IZ" (qubit 0 first)."""
    return "".join(LETTERS[((x[0] >> q) & 1, (z[0] >> q) & 1)] for q in range(n))


def compile_tape(n, gates, observable):
    """gates: [(name, qubit, param index)] in circuit order.
    observable: [(word string, coefficient)]."""
    def bits(word, letters):
        return [sum(1 << q for q, ch in enumerate(word) if ch in letters)]
    obs = [(bits(w, "XY"), bits(w, "ZY"), c, [], []) for w, c in observable]
    n_boxes, init, steps, pairs, readout, words = pprop_rs.tape_debug(
        n,
        [_GATE_KIND[g] for g, _, _ in gates],
        [q for _, q, _ in gates],
        [-1] * len(gates),
        [p for _, _, p in gates],
        [None] * len(gates),
        obs,
    )
    return dict(n_boxes=n_boxes, init=init, steps=steps, pairs=pairs, readout=readout,
                words=[word_str(x, z, n) for x, z in words])


def test_example_by_hand():
    # RX(θ0) q0, RY(θ1) q1, RX(θ2) q1, observable Z on q1. Backwards:
    #   RX(θ2) q1: IZ -> cos·IZ + sin·IY       pair [IZ, IY]
    #   RY(θ1) q1: IZ -> cos·IZ − sin·IX       pair [IX, IZ]  (X -> cos·X + sin·Z)
    #              IY commutes with RY
    #   RX(θ0) q0: every word has I on q0      no step
    t = compile_tape(2, [("RX", 0, 0), ("RY", 1, 1), ("RX", 1, 2)], [("IZ", 1.0)])
    assert t["words"] == ["IZ", "IY", "IX"]
    assert t["n_boxes"] == 3
    assert t["init"] == [(0, 1.0)]
    assert t["steps"] == [(2, 0, 1), (1, 1, 2)]
    assert t["pairs"] == [(0, 1), (2, 0)]
    assert t["readout"] == [0]


def test_pair_seen_from_both_sides_is_written_once():
    # After RX(θ1), both IZ and IY exist; RX(θ0) on the same qubit meets the
    # pair from IZ and from IY, and must write it once.
    t = compile_tape(1, [("RX", 0, 0), ("RX", 0, 1)], [("Z", 1.0)])
    assert t["words"] == ["Z", "Y"]
    assert t["steps"] == [(1, 0, 1), (0, 1, 2)]
    assert t["pairs"] == [(0, 1), (0, 1)]


def test_observable_word_listed_twice_gets_one_box():
    t = compile_tape(1, [("RX", 0, 0)], [("Z", 0.5), ("Z", 0.25)])
    assert t["init"] == [(0, 0.75)]


@pytest.mark.parametrize("seed", range(20))
def test_random_rotation_circuits(seed):
    rng = random.Random(seed)
    n = rng.randint(1, 4)
    gates = [(rng.choice(["RX", "RY", "RZ"]), rng.randrange(n), p) for p in range(rng.randint(1, 12))]
    obs_word = "".join(rng.choice("IZ") for _ in range(n - 1)) + "Z"
    t = compile_tape(n, gates, [(obs_word, 1.0)])
    words = t["words"]
    assert len(set(words)) == len(words) == t["n_boxes"]

    by_param = {p: (g, q) for g, q, p in gates}
    for p, first, last in t["steps"]:
        gate, q = by_param[p]
        step_pairs = t["pairs"][first:last]
        boxes = [b for pair in step_pairs for b in pair]
        assert len(boxes) == len(set(boxes)), "a box appears twice in one step"
        for a, b in step_pairs:
            wa, wb = words[a], words[b]
            # The two words differ only on the gate's qubit, with the letters
            # of the pair order [a, b].
            assert [i for i in range(n) if wa[i] != wb[i]] == [q]
            assert (wa[q], wb[q]) == PAIR_LETTERS[gate]
    # Gates are walked in reverse, so the steps come in reverse parameter order.
    order = [p for p, _, _ in t["steps"]]
    assert order == sorted(order, reverse=True)
    assert t["readout"] == [b for b, w in enumerate(words) if set(w) <= {"I", "Z"}]


def test_gates_not_supported_yet():
    with pytest.raises(ValueError, match="cannot compile gate kind"):
        pprop_rs.tape_debug(2, [_GATE_KIND["CNOT"]], [0], [1], [-1], [None], [([0], [2], 1.0, [], [])])
    with pytest.raises(ValueError, match="fixed angle"):
        pprop_rs.tape_debug(1, [_GATE_KIND["RX"]], [0], [-1], [-1], [0.3], [([0], [1], 1.0, [], [])])
