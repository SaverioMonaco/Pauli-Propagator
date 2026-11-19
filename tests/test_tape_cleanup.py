"""The cleanup at the end of compiling: every update whose result never reaches
the readout is removed, and the boxes left are renumbered densely.

compile_tapes (and so Propagator) always cleans up, so every value and
gradient test of the tape also tests that the cleanup changes nothing. These
tests look at the structure, with tape_debug(cleanup=True) against the raw tape.
"""
import random

import numpy as np
import pytest

import pprop_rs
from pprop.propagator import _GATE_KIND
from test_tape_compile import word_str
from test_tape_eval import obs_spec, with_params
from test_tape_grad import every_gate_case


def debug(n, gates, observable, cleanup, **kw):
    gates_p, _ = with_params(gates)
    wires = [w if isinstance(w, tuple) else (w, -1) for _, w, _, _ in gates_p]
    d = pprop_rs.tape_debug(
        n, [_GATE_KIND[g] for g, _, _, _ in gates_p], [w0 for w0, _ in wires], [w1 for _, w1 in wires],
        [p for _, _, p, _ in gates_p], [f for _, _, _, f in gates_p], obs_spec(observable),
        cleanup=cleanup, **kw)
    d["words"] = [word_str(x, z, n) for x, z in d["words"]]
    return d


def boxes_used(d):
    used = {b for b, _ in d["init"]} | set(d["readout"]) | set(d["negs"]) | {b for _, b in d["singles"]}
    used |= {b for pair in d["pairs"] for b in pair}
    used |= {m for _, members in d["blocks"] for m in members if m >= 0}
    return used


def test_two_qubit_example_loses_the_dead_pair():
    # RX(θ0) q0, RX(θ1) q1, CNOT, RX(θ2) q0, RX(θ3) q1, <Z1>: the last step pairs
    # ZZ with YZ and ZY with YY. Only ZZ is read out, and nothing reads ZY or
    # YY afterwards, so the pair (ZY, YY) and the box YY go.
    gates = [("RX", 0), ("RX", 1), ("CNOT", (0, 1)), ("RX", 0), ("RX", 1)]
    raw = debug(2, gates, [("IZ", 1.0)], cleanup=False)
    clean = debug(2, gates, [("IZ", 1.0)], cleanup=True)
    assert raw["words"] == ["ZZ", "ZY", "YZ", "YY"] and len(raw["pairs"]) == 4
    assert clean["words"] == ["ZZ", "ZY", "YZ"] and clean["pairs"] == [(0, 1), (0, 1), (0, 2)]
    assert clean["readout"] == [0]


def test_observable_that_never_reaches_the_readout_leaves_nothing():
    # X on q0 with gates only on q1: X stays on q0, and <X> in |0> is 0.
    clean = debug(2, [("RX", 1), ("RY", 1)], [("XI", 1.0)], cleanup=True)
    assert clean["init"] == [] and clean["readout"] == [] and clean["steps"] == [] and clean["n_boxes"] == 0


@pytest.mark.parametrize("seed", range(30))
def test_cleanup_only_removes_and_renumbers_densely(seed):
    n, gates, observable, _ = every_gate_case(seed)
    rng = random.Random(seed)
    kw = dict(k1=rng.choice([-1, 1, n]), use_dead_qubit_pruner=rng.random() < 0.5,
              k2=rng.choice([-1, 2, 5]), coeff_threshold=rng.choice([-1.0, 0.05]))
    raw = debug(n, gates, observable, cleanup=False, **kw)
    clean = debug(n, gates, observable, cleanup=True, **kw)
    assert len(clean["pairs"]) <= len(raw["pairs"]) and len(clean["blocks"]) <= len(raw["blocks"])
    assert len(clean["singles"]) <= len(raw["singles"]) and len(clean["negs"]) <= len(raw["negs"])
    assert clean["n_boxes"] <= raw["n_boxes"] and len(clean["steps"]) <= len(raw["steps"])
    # Dense: the boxes used are exactly 0 .. n_boxes - 1.
    assert boxes_used(clean) == set(range(clean["n_boxes"]))
    # The readout still names the same words.
    assert sorted(clean["words"][b] for b in clean["readout"]) == sorted(raw["words"][b] for b in raw["readout"])
    # No step is left empty.
    for kind, _, first, last in clean["steps"]:
        if kind == "negate":
            assert last > first
