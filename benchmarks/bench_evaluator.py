"""
Evaluator benchmark on the repo's own circuits.

The circuit is the 2D transverse-field Ising model and the shell-brick ansatz
used by ``scripts/vqe`` and by the pruning section of
``notebooks/extra/new_stuff.ipynb``. Two variants are measured:

  plain              the ansatz as written
  fixed-value layer  with a non-parametrised RY(pi) layer

The baseline is loaded from a git ref rather than pasted in here, so the
comparison always reflects the repository's own history.

    python benchmarks/bench_evaluator.py [--side 6] [--layers 3] [--repeats 15]
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import time

import numpy as np
import pennylane as qml

from pprop import Propagator
from pprop.propagator import evaluator


def load_baseline(ref: str = "main"):
    """
    Load the evaluator as it exists at ``ref`` (default ``main``), so the
    comparison baseline comes from this repository's own history rather than
    from a copy pasted in here.

    Returns ``None`` if ``ref`` doesn't resolve or has no evaluator.
    """
    try:
        src = subprocess.run(
            ["git", "show", f"{ref}:src/pprop/propagator/utils.py"],
            capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    # Loaded standalone, so the package-relative import has to be absolute.
    src = src.replace("from ..pauli.sentence import", "from pprop.pauli.sentence import")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(src)
        path = fh.name

    name = "pprop_baseline_utils"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"  (baseline {ref} did not load: {exc})")
        return None
    return mod if hasattr(mod, "make_sparse_evaluator") else None


def hamiltonian(side: int, J: float = 1.0, h: float = 1.0) -> qml.Hamiltonian:
    """2D transverse-field Ising, as in scripts/vqe/ising2d.py."""
    coeffs, obs = [], []
    n = side * side
    for x in range(side):
        for y in range(side):
            i = x * side + y
            if y < side - 1:
                coeffs.append(-J / n)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(x * side + y + 1))
            if x < side - 1:
                coeffs.append(-J / n)
                obs.append(qml.PauliZ(i) @ qml.PauliZ((x + 1) * side + y))
    for i in range(n):
        coeffs.append(-h / n)
        obs.append(qml.PauliX(i))
    return qml.Hamiltonian(coeffs, obs)


def make_circuit(side: int, layers: int, fixed_layer: bool):
    """The shell-brick ansatz from notebooks/extra/new_stuff.ipynb."""
    n = side * side
    ham = hamiltonian(side)

    def circuit(params):
        index = 0
        for q in range(n):
            qml.RX(params[index], wires=q)
            index += 1
        for d in range(layers):
            y0 = 0 if d % 2 == 0 else 1
            for x in range(side):
                for y in range(y0, side - 1, 2):
                    qml.CNOT(wires=[x * side + y, x * side + y + 1])
            for q in range(n):
                qml.RY(params[index], wires=q)
                index += 1
            if fixed_layer and d == 0:
                # A non-parametrised gate: a fixed angle, not a trainable index.
                # sin(pi) == 0 makes every term carrying it identically zero.
                for q in range(n):
                    qml.RY(np.pi, wires=q)
        return qml.expval(ham)

    return circuit


def bench(fn, args, repeats: int) -> float:
    fn(*args)
    t = time.perf_counter()
    for _ in range(repeats):
        fn(*args)
    return (time.perf_counter() - t) / repeats * 1e3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--side", type=int, default=6)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--baseline", default="0511b27",
                   help="git ref to compare against (default: 0511b27)")
    a = p.parse_args()

    print(f"{a.side}x{a.side} transverse-field Ising, {a.layers} brick layers")

    base = load_baseline(a.baseline)
    print(f"baseline: {a.baseline} "
          f"({'loaded' if base else 'unavailable, skipping old-vs-new'})\n")

    rng = np.random.default_rng(0)

    for label, fixed in (("plain", False), ("fixed-value layer", True)):
        prop = Propagator(make_circuit(a.side, a.layers, fixed))
        prop.propagate(use_dead_qubit_pruner=True, use_xy_weight_pruner=True)
        expr = prop.exprs[0]
        ip = prop._internal_num_params

        theta = rng.uniform(-np.pi, np.pi, ip)
        for value, slot in prop._fixed_value_slots.items():
            theta[slot] = value
        sins, coss = np.sin(theta), np.cos(theta)

        _, eg_new = evaluator.make_sparse_evaluator(expr, ip)
        t_new = bench(eg_new, (sins, coss), a.repeats)

        print(f"{label}: {prop.num_params} params, {len(expr)} terms")
        if base is not None:
            _, eg_base = base.make_sparse_evaluator(expr, ip)
            t_base = bench(eg_base, (sins, coss), a.repeats)
            v_b, g_b = eg_base(sins, coss)
            v_n, g_n = eg_new(sins, coss)
            # Only the trainable range is meaningful: Propagator.eval_and_grad
            # slices the gradient to [:num_params] before the caller sees it.
            t_ = prop.num_params
            g_b, g_n = g_b[:t_], g_n[:t_]
            print(f"  {a.baseline:<14} : {t_base:8.2f} ms")
            print(f"  this branch    : {t_new:8.2f} ms   ({t_base / t_new:.1f}x)")
            print(f"  agreement      : value "
                  f"{abs(v_n - v_b) / max(abs(v_b), 1e-30):.1e}, gradient "
                  f"{np.abs(g_n - g_b).max() / max(np.abs(g_b).max(), 1e-30):.1e}")
        else:
            print(f"  eval_and_grad  : {t_new:8.2f} ms")
        print()


if __name__ == "__main__":
    main()
