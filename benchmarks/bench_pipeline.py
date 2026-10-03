"""Compare complete pipelines with each revision's own expression and extension.

Requires maturin and the project's Python dependencies in the active Python
environment. The baseline is exported from git and built separately; no branch
is checked out and no extension is installed into the active environment.

Examples (run from the repository root)::

    python benchmarks/bench_pipeline.py --baseline a1f767d --side 6
    python benchmarks/bench_pipeline.py --baseline HEAD^ --compare-numpy
    python benchmarks/bench_pipeline.py --baseline HEAD^ --threads 8

Each propagation sample runs in a fresh process. Peak RSS includes Python and
its imported dependencies, but excludes the extension build. Propagation time
includes evaluator compilation; the native-call column includes conversion of
arguments/results at the Python boundary. Evaluation times are warm averages.
Use identical machine/load conditions and multiple samples for performance
claims. ``bench_evaluator.py`` separately compares evaluators on one expression.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

import numpy as np


def worker(args):
    # Imports resolve against the chosen revision and its extracted wheel.
    import pprop_rs
    from pprop import Propagator
    from bench_evaluator import make_circuit
    try:
        from pprop.propagator import evaluator
    except ImportError:
        from pprop.propagator import utils as evaluator

    if args.numpy:
        if not hasattr(evaluator, "Evaluator"):
            raise RuntimeError("--compare-numpy requires a revision with the Rust evaluator")
        evaluator.Evaluator = None

    kernel_times = []
    original = pprop_rs.propagate_batch

    def timed_kernel(*a, **kw):
        start = time.perf_counter()
        result = original(*a, **kw)
        kernel_times.append(time.perf_counter() - start)
        return result

    pprop_rs.propagate_batch = timed_kernel
    prop = Propagator(make_circuit(args.side, args.layers, args.fixed))
    options = dict(use_dead_qubit_pruner=True, use_xy_weight_pruner=True)
    if "n_threads" in inspect.signature(prop.propagate).parameters:
        options["n_threads"] = args.threads
    elif args.threads != 1:
        raise RuntimeError("This revision does not support --threads other than 1")
    start = time.perf_counter()
    prop.propagate(**options)
    propagation = time.perf_counter() - start
    theta = np.random.default_rng(0).uniform(-np.pi, np.pi, prop.num_params)
    value, gradient = prop.eval_and_grad(theta)
    start = time.perf_counter()
    for _ in range(args.repeats):
        prop.eval_and_grad(theta)
    evaluation = (time.perf_counter() - start) / args.repeats
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mib = rss / (1024 ** 2 if sys.platform == "darwin" else 1024)
    print(json.dumps(dict(
        terms=sum(map(len, prop.exprs)), parameters=prop.num_params,
        propagation_ms=propagation * 1000,
        native_call_ms=sum(kernel_times) * 1000,
        evaluation_ms=evaluation * 1000, peak_rss_mib=rss_mib,
        value=np.asarray(value).tolist(), gradient=np.asarray(gradient).tolist(),
        evaluator="rust" if getattr(evaluator, "Evaluator", None) is not None else "numpy",
    )))


def build_extension(source, destination, offline, cache=None):
    if cache is not None:
        digest = hashlib.sha256(sys.version.encode())
        for path in sorted((source / "native/pprop_rs").rglob("*.rs")):
            digest.update(path.relative_to(source).as_posix().encode())
            digest.update(path.read_bytes())
        for name in ("Cargo.toml", "Cargo.lock", "pyproject.toml", "python/pprop_rs/__init__.py"):
            digest.update((source / "native/pprop_rs" / name).read_bytes())
        destination = cache.resolve() / digest.hexdigest()
    wheel_dir = destination / "wheels"
    wheel_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Every revision gets its own target: git archive timestamps can otherwise
    # cause Cargo to reuse a different revision's already-built extension.
    env["CARGO_TARGET_DIR"] = str(destination / "target")
    command = [sys.executable, "-m", "maturin", "build", "--release", "--locked",
               "--manifest-path", str(source / "native/pprop_rs/Cargo.toml"),
               "--interpreter", sys.executable, "--out", str(wheel_dir)]
    if offline:
        command.append("--offline")
    subprocess.run(command, env=env, check=True, stdout=sys.stderr)
    package = destination / "package"
    with zipfile.ZipFile(next(wheel_dir.glob("*.whl"))) as wheel:
        wheel.extractall(package)
    return package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="a1f767d",
                        help="git revision to build separately (default: pre-series a1f767d)")
    parser.add_argument("--side", type=int, default=6)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--baseline-threads", type=int, default=1)
    parser.add_argument("--compare-numpy", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--build-cache", type=Path, help="Reuse builds keyed by native source and Python version")
    parser.add_argument("--output", type=Path, help="Save raw samples and revision metadata as JSON")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--fixed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--numpy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.side, args.repeats, args.samples, args.threads, args.baseline_threads) < 1 or args.layers < 0:
        parser.error("sizes, repetitions and thread counts must be positive; layers may be zero")
    if args.worker:
        worker(args)
        return

    root = Path(__file__).resolve().parents[1]
    baseline_commit = subprocess.check_output(
        ["git", "rev-parse", args.baseline + "^{commit}"], cwd=root, text=True).strip()
    current_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root))
    report = dict(baseline=baseline_commit, current=current_commit, current_dirty=dirty,
                  python=sys.version, machine=platform.platform(),
                  settings={k: v for k, v in vars(args).items() if k not in ("output", "build_cache")}, samples=[])
    print(f"{args.side}x{args.side}, {args.layers} layers; median of {args.samples} fresh processes")
    print(f"baseline {baseline_commit}; current {current_commit}{' (modified)' if dirty else ''}")
    print("case / implementation        terms  propagate ms  native ms  eval+grad ms  peak RSS MiB")
    with tempfile.TemporaryDirectory(prefix="pprop-benchmark-") as tmp:
        staging = Path(tmp)
        baseline = staging / "baseline"
        baseline.mkdir()
        archive = subprocess.check_output(["git", "archive", baseline_commit], cwd=root)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(baseline, filter="data")
        packages = {"baseline": build_extension(baseline, staging / "base-build", args.offline, args.build_cache),
                    "current": build_extension(root, staging / "current-build", args.offline, args.build_cache)}
        variants = [("baseline", baseline, args.baseline_threads, False),
                    ("current", root, args.threads, False)]
        if args.compare_numpy:
            variants.append(("current numpy", root, args.threads, True))
        for fixed in (False, True):
            reference = None
            for label, source, threads, numpy in variants:
                rows = []
                for sample in range(args.samples):
                    command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                               "--side", str(args.side), "--layers", str(args.layers),
                               "--repeats", str(args.repeats), "--threads", str(threads)]
                    if fixed:
                        command.append("--fixed")
                    if numpy:
                        command.append("--numpy")
                    env = os.environ.copy()
                    package = packages["baseline" if label == "baseline" else "current"]
                    env["PYTHONPATH"] = os.pathsep.join([str(source / "src"), str(package)])
                    result = subprocess.check_output(command, env=env, cwd=source, text=True)
                    row = json.loads(result.splitlines()[-1])
                    if reference is None:
                        reference = row
                    for field in ("value", "gradient"):
                        if np.shape(row[field]) != np.shape(reference[field]) or not np.allclose(
                                row[field], reference[field], rtol=1e-9, atol=1e-10):
                            raise RuntimeError(f"{label}: {field} disagrees with baseline")
                    rows.append(row)
                    report["samples"].append(dict(case="fixed" if fixed else "plain",
                                                   implementation=label, sample=sample, threads=threads, **row))
                summary = {key: statistics.median(r[key] for r in rows)
                           for key in ("terms", "propagation_ms", "native_call_ms", "evaluation_ms", "peak_rss_mib")}
                name = ("fixed" if fixed else "plain") + " / " + label
                print(f"{name:27} {summary['terms']:7.0f} {summary['propagation_ms']:13.2f} "
                      f"{summary['native_call_ms']:10.2f} {summary['evaluation_ms']:13.3f} "
                      f"{summary['peak_rss_mib']:13.1f}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
