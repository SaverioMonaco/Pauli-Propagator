"""The tape compiler (pprop_rs.tape_debug), checked before tapes can be evaluated.

The tape compiles every gate pprop supports: RX, RY, RZ (trainable or fixed),
T, the Clifford gates (H, S, SX, CNOT, CY, CZ, SWAP) and the controlled
rotations (CRX, CRY, CRZ), with the k1 truncation and the two exact pruners.
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


def compile_tape(n, gates, observable, k1=-1, dead=False, xy=False, k2=-1, sine=False):
    """gates: [(name, wires, param index)] in circuit order; wires is a qubit or
    a (control, target) pair; param is -1 for a Clifford gate.
    observable: [(word string, coefficient)].
    Steps come back as ("rotate", param, first, last) or ("negate", -1, first, last)."""
    def bits(word, letters):
        return [sum(1 << q for q, ch in enumerate(word) if ch in letters)]
    wires = [w if isinstance(w, tuple) else (w, -1) for _, w, _ in gates]
    obs = [(bits(w, "XY"), bits(w, "ZY"), c, [], []) for w, c in observable]
    d = pprop_rs.tape_debug(
        n,
        [_GATE_KIND[g] for g, _, _ in gates],
        [w0 for w0, _ in wires],
        [w1 for _, w1 in wires],
        [p for _, _, p in gates],
        [None] * len(gates),
        obs,
        k1,
        dead,
        xy,
        k2,
        sine,
    )
    d["words"] = [word_str(x, z, n) for x, z in d["words"]]
    return d


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
    assert t["steps"] == [("rotate", 2, 0, 1), ("rotate", 1, 1, 2)]
    assert t["pairs"] == [(0, 1), (2, 0)]
    assert t["readout"] == [0]


def test_pair_seen_from_both_sides_is_written_once():
    # After RX(θ1), both IZ and IY exist; RX(θ0) on the same qubit meets the
    # pair from IZ and from IY, and must write it once.
    t = compile_tape(1, [("RX", 0, 0), ("RX", 0, 1)], [("Z", 1.0)])
    assert t["words"] == ["Z", "Y"]
    assert t["steps"] == [("rotate", 1, 0, 1), ("rotate", 0, 1, 2)]
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
    for kind, p, first, last in t["steps"]:
        assert kind == "rotate"
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
    order = [p for _, p, _, _ in t["steps"]]
    assert order == sorted(order, reverse=True)
    assert t["readout"] == [b for b, w in enumerate(words) if set(w) <= {"I", "Z"}]


def test_cnot_renames_boxes():
    # The two-qubit example: RX(θ0) q0, RX(θ1) q1, CNOT(0 -> 1), RX(θ2) q0, RX(θ3) q1, <Z1>.
    # Backwards: RX(θ3) pairs IZ with IY; RX(θ2) touches nothing; the CNOT
    # renames IZ -> ZZ and IY -> ZY (no sign, no step); RX(θ1) mixes the pair
    # (ZZ, ZY); RX(θ0) pairs ZZ with YZ and ZY with YY.
    gates = [("RX", 0, 0), ("RX", 1, 1), ("CNOT", (0, 1), -1), ("RX", 0, 2), ("RX", 1, 3)]
    t = compile_tape(2, gates, [("IZ", 1.0)])
    assert t["words"] == ["ZZ", "ZY", "YZ", "YY"]
    assert t["steps"] == [("rotate", 3, 0, 1), ("rotate", 1, 1, 2), ("rotate", 0, 2, 4)]
    assert t["pairs"] == [(0, 1), (0, 1), (0, 2), (1, 3)]
    assert t["negs"] == []
    assert t["readout"] == [0]


def test_clifford_minus_sign_is_a_negate_step():
    # H turns Y into −Y: the box keeps its word and gets negated.
    t = compile_tape(1, [("Hadamard", 0, -1)], [("Y", 1.0)])
    assert t["words"] == ["Y"]
    assert t["steps"] == [("negate", -1, 0, 1)] and t["negs"] == [0]
    # H turns X into Z: renamed, no sign, no step.
    t = compile_tape(1, [("Hadamard", 0, -1)], [("X", 1.0)])
    assert t["words"] == ["Z"] and t["steps"] == [] and t["readout"] == [0]


def test_swap_exchanges_names_in_one_go():
    # Both words exist before the SWAP and swap names: renaming one at a time
    # would make the first new name collide with the second old one.
    t = compile_tape(2, [("SWAP", (0, 1), -1)], [("XZ", 1.0), ("ZX", 2.0)])
    assert t["words"] == ["ZX", "XZ"]
    assert t["init"] == [(0, 1.0), (1, 2.0)]
    assert t["steps"] == []


def test_k1_drops_a_word_a_cnot_makes_too_heavy():
    # RX(θ0) q1, CNOT(0 -> 1), RX(θ1) q0, <Z0>, k1 = 1. Backwards: RX(θ1)
    # pairs ZI with YI; the CNOT leaves ZI alone but turns YI into YX
    # (weight 2 > k1), so YI's box is dropped; RX(θ0) then touches nothing.
    gates = [("RX", 1, 0), ("CNOT", (0, 1), -1), ("RX", 0, 1)]
    t = compile_tape(2, gates, [("ZI", 1.0)], k1=1)
    assert t["words"] == ["ZI", "YI"] and t["alive"] == [True, False]
    assert t["steps"] == [("rotate", 1, 0, 1)]
    assert t["readout"] == [0]
    # Without k1 the word lives on as YX (RX(θ0) on q1 leaves its X alone).
    t = compile_tape(2, gates, [("ZI", 1.0)])
    assert t["words"] == ["ZI", "YX"] and all(t["alive"])


def test_observable_cut_after_its_first_gate():
    # ZZ is too heavy for k1 = 1, but the first gate touching it is a CNOT
    # that turns it into IZ: it is kept, as in the sum-of-terms engine.
    t = compile_tape(2, [("CNOT", (0, 1), -1)], [("ZZ", 1.0)], k1=1)
    assert t["words"] == ["IZ"] and t["alive"] == [True] and t["readout"] == [0]
    # A rotation keeps its weight: ZZ is dropped right after it.
    t = compile_tape(2, [("RX", 0, 0)], [("ZZ", 1.0)], k1=1)
    assert t["alive"] == [False] and t["readout"] == [] and t["steps"] == []
    # No gate touches the observable: nothing is cut.
    t = compile_tape(3, [("RX", 2, 0)], [("ZZI", 1.0)], k1=1)
    assert t["alive"] == [True] and t["readout"] == [0]


def test_dead_qubit_pruner():
    # RX(θ0) q0, RX(θ1) q1, <ZZ>. Backwards: RX(θ1) pairs ZZ with ZY. Then
    # only RX(θ0) on q0 is left: ZY has a Y on q1, which no remaining gate
    # touches, so it can never reach the readout and is dropped before the gate.
    gates = [("RX", 0, 0), ("RX", 1, 1)]
    t = compile_tape(2, gates, [("ZZ", 1.0)], dead=True)
    assert t["words"] == ["ZZ", "ZY", "YZ"] and t["alive"] == [True, False, True]
    assert t["pairs"] == [(0, 1), (0, 2)]
    # Without the pruner ZY is paired with YY too.
    t = compile_tape(2, gates, [("ZZ", 1.0)])
    assert t["words"] == ["ZZ", "ZY", "YZ", "YY"] and t["pairs"] == [(0, 1), (0, 2), (1, 3)]


def test_xy_weight_pruner():
    # <YY> through one gate on q0: a gate removes at most one X/Y, and YY has
    # two, so it is dropped before RX(θ0) acts on it.
    t = compile_tape(2, [("RX", 0, 0)], [("YY", 1.0)], xy=True)
    assert t["alive"] == [False] and t["steps"] == [] and t["readout"] == []
    t = compile_tape(2, [("RX", 0, 0)], [("YY", 1.0)])
    assert t["words"] == ["YY", "ZY"] and t["steps"] == [("rotate", 0, 0, 1)]


def test_fixed_angle_step():
    # A fixed RX(0.3): the same pair as a trainable one, in a "fixed" step.
    out = pprop_rs.tape_debug(1, [_GATE_KIND["RX"]], [0], [-1], [-1], [0.3], [([0], [1], 1.0, [], [])])
    steps, pairs = out["steps"], out["pairs"]
    assert steps == [("fixed", -1, 0, 1)] and pairs == [(0, 1)]


def test_debug_fields():
    # tape_debug returns a dict; blocks stay empty until a gate compiles to them.
    d = pprop_rs.tape_debug(1, [_GATE_KIND["RX"]], [0], [-1], [0], [None], [([0], [1], 1.0, [], [])])
    # New fields may be added over time; these are the ones tests rely on.
    assert {"n_boxes", "init", "steps", "pairs", "negs", "blocks", "block_progs",
            "readout", "words", "alive", "k2", "sine_order"} <= set(d)
    assert d["k2"] == -1 and d["sine_order"] is False
    assert d["blocks"] == [] and d["block_progs"] == []


def test_k2_drops_words_that_can_only_hold_zeros():
    # RX(θ0), RX(θ1) on <Z>, k2 = 1 (frequency): after RX(θ1), Z and Y each have
    # one factor (low 1); RX(θ0) adds one more to everything, so both words
    # could only hold order 2 > k2: dropped, and the readout is empty.
    t = compile_tape(1, [("RX", 0, 0), ("RX", 0, 1)], [("Z", 1.0)], k2=1)
    assert t["words"] == ["Z", "Y"] and t["alive"] == [False, False] and t["readout"] == []


def test_k2_singles_in_sine_mode():
    # Sine mode, k2 = 0: Y would only get Z's sin part (one sin), so it is never
    # made, and Z keeps its cos part at both gates: two steps with a single each.
    t = compile_tape(1, [("RX", 0, 0), ("RX", 0, 1)], [("Z", 1.0)], k2=0, sine=True)
    assert t["words"] == ["Z"] and t["n_boxes"] == 1 and t["pairs"] == []
    assert t["singles"] == [(0, 0), (1, 0)] and t["readout"] == [0]


def test_t_gate_is_a_fixed_rz_pair():
    # T: X -> X/√2 − Y/√2, so the pair is [Y, X] as for RZ, in a fixed step.
    out = pprop_rs.tape_debug(1, [_GATE_KIND["T"]], [0], [-1], [-1], [None], [([1], [0], 1.0, [], [])])
    steps, pairs, words = out["steps"], out["pairs"], out["words"]
    assert steps == [("fixed", -1, 0, 1)]
    assert [word_str(x, z, 1) for x, z in words] == ["X", "Y"] and pairs == [(1, 0)]
    # T commutes with Z: no step.
    out = pprop_rs.tape_debug(1, [_GATE_KIND["T"]], [0], [-1], [-1], [None], [([0], [1], 1.0, [], [])])
    assert out["steps"] == []


def test_controlled_rotation_is_a_block():
    # CRX(0 -> 1) on IZ: IZ belongs to CRX's group of 4 (IZ, IY, ZZ, ZY), so
    # one block whose other members are new boxes.
    t = compile_tape(2, [("CRX", (0, 1), 0)], [("IZ", 1.0)])
    assert t["steps"] == [("block", 0, 0, 1)]
    ((prog, members),) = t["blocks"]
    assert [t["words"][m] for m in members] == ["IZ", "IY", "ZZ", "ZY"]
    assert t["block_progs"][prog][0] == 4
    # With k1 = 1, ZZ and ZY are too heavy: those members are NONE (-1).
    t = compile_tape(2, [("CRX", (0, 1), 0)], [("IZ", 1.0)], k1=1)
    ((prog, members),) = t["blocks"]
    assert [t["words"][m] for m in members[:2]] == ["IZ", "IY"] and members[2:] == [-1, -1]
    # CRX on XI: a 2-member group (XI, YX).
    t = compile_tape(2, [("CRX", (0, 1), 0)], [("XI", 1.0)])
    ((prog, members),) = t["blocks"]
    assert sorted(t["words"][m] for m in members) == ["XI", "YX"] and t["block_progs"][prog][0] == 2


def test_fixed_controlled_rotation_is_a_fixed_block():
    d = pprop_rs.tape_debug(2, [_GATE_KIND["CRY"]], [0], [1], [-1], [0.8], [([0], [2], 1.0, [], [])])
    assert d["steps"] == [("block_fixed", -1, 0, 1)]
