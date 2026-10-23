"""
Train the shell-brick circuit (model.py) to generate binarised MNIST digits.

Loss
----
The Gaussian-kernel MMD between the circuit's bitstring distribution and the
data can be written as a weighted sum over PauliZ strings. We estimate it
stochastically: each *round* draws a batch of random Z strings ``O_k`` (each
qubit included with probability ``p(sigma)``) and minimises

    L(theta) = sum_k  w_k * (<O_k>_theta - mu_k)^2

where ``mu_k`` is the same moment measured on the training images and ``w_k``
is how often ``O_k`` was drawn. ``<O_k>_theta`` comes from a pprop
Propagator (truncated with k1/k2), and Adam runs ``--num_steps`` steps on it.

Output
------
    <path>/<run name>/<NNN>/result.json   trained params, losses, test MMD, config
    <path>/<run name>/<NNN>/circuit.qasm  the trained circuit, ready to deploy

Usage
-----
    python train.py --side 8 --num_obs 1000 --num_rounds 100 \\
                    --num_steps 1000 --lr 5e-5 --k1 12
    python train.py --side 4 --num_rounds 3 --num_steps 50   # quick smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pprop import Propagator, mnist
from pprop.optimization import adam
from pprop.qasm import save_qasm

import model


# ---------------------------------------------------------------------------
# Stochastic MMD batch
# ---------------------------------------------------------------------------

def sample_z_strings(num_qubits: int, sigma: float, n_obs: int, rng) -> dict[tuple, int]:
    """Draw ``n_obs`` distinct non-empty Z strings; return {qubits: times drawn}.

    Each qubit is included independently with probability ``t / (1 + t)``,
    ``t = tanh(1 / (4 sigma^2))``, which is the Pauli-Z expansion of a
    Gaussian kernel of bandwidth ``sigma`` on bitstrings.
    """
    t = np.tanh(1.0 / (4.0 * sigma ** 2))
    p = t / (1.0 + t)
    counts: dict[tuple, int] = {}
    while len(counts) < n_obs:
        qubits = tuple(int(q) for q in np.flatnonzero(rng.random(num_qubits) < p))
        if qubits:
            counts[qubits] = counts.get(qubits, 0) + 1
    return counts


def make_batch(side, sigma, X, n_obs, k1, k2, rng):
    """Propagate one random batch; return (propagator, data moments, weights)."""
    counts = sample_z_strings(side * side, sigma, n_obs, rng)
    subsets = list(counts)
    weights = np.array([counts[s] for s in subsets], dtype=float)

    observables = [model.z_string(s) for s in subsets]
    prop = Propagator(model.circuit(side, observables), k1=k1, k2=k2)
    prop.propagate(use_xy_weight_pruner=True, use_dead_qubit_pruner=True)

    return prop, mnist.z_moments(X, subsets), weights / weights.sum()


def batch_loss(prop, mu, w, params) -> float:
    return float(np.dot(w, (prop(params) - mu) ** 2))


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train(side, sigma, X, params, *, n_obs, n_rounds, steps, lr, k1, k2, seed, log=None):
    """Run ``n_rounds`` rounds of Adam, each on a fresh observable batch."""
    rng = np.random.default_rng(seed)
    losses, round_losses = [], []

    for r in range(n_rounds):
        prop, mu, w = make_batch(side, sigma, X, n_obs, k1, k2, rng)
        result = adam(
            lambda f: float(np.dot(w, (f - mu) ** 2)),
            prop, params,
            lr=lr, num_steps=steps, print_every=0,
            grad_L=lambda f: 2.0 * w * (f - mu),
        )
        params = np.asarray(result["params"])
        losses += list(result["loss_history"])
        round_losses.append(float(np.mean(result["loss_history"])))
        print(f"round {r + 1:4d}/{n_rounds}   mean loss {round_losses[-1]:.3e}")
        if log:
            log({"round": r + 1, "loss/round_mean": round_losses[-1]}, step=(r + 1) * steps)

    return params, losses, round_losses


def evaluate(side, sigma, X, params, *, n_obs, n_batches, k1, k2, seed):
    """Batch MMD on held-out data: mean and std over ``n_batches`` random batches."""
    rng = np.random.default_rng(seed)
    values = [batch_loss(*make_batch(side, sigma, X, n_obs, k1, k2, rng), params)
              for _ in range(n_batches)]
    return float(np.mean(values)), float(np.std(values))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--side", type=int, default=8, help="grid side; num_qubits = side^2")
    p.add_argument("--digits", type=int, nargs="+", default=list(range(10)), help="digit classes to learn")
    p.add_argument("--max_train", type=int, default=5000, help="training images used")
    p.add_argument("--num_obs", type=int, default=20, help="Z strings per round")
    p.add_argument("--num_rounds", type=int, default=100, help="rounds (fresh batch each)")
    p.add_argument("--num_steps", type=int, default=1000, help="Adam steps per round")
    p.add_argument("--lr", type=float, default=0.1, help="Adam learning rate")
    p.add_argument("--k1", type=int, default=6, help="Pauli-weight truncation")
    p.add_argument("--k2", type=int, default=None, help="frequency truncation (default: none)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--path", type=str, default="results", help="output root")
    p.add_argument("--wandb", type=str, default=None, metavar="PROJECT",
                   help="log to this Weights & Biases project (off by default)")
    return p.parse_args()


def next_run_dir(root: Path) -> Path:
    """``root/000``, ``root/001``, ... : first index not yet taken."""
    root.mkdir(parents=True, exist_ok=True)
    idx = 0
    while (root / f"{idx:03d}").exists():
        idx += 1
    out = root / f"{idx:03d}"
    out.mkdir()
    return out


def main():
    args = parse_args()
    if any(not 0 <= d <= 9 for d in args.digits):
        raise SystemExit(f"--digits must be in 0-9, got {args.digits}")

    digits = "".join(map(str, sorted(args.digits)))
    name = f"side{args.side}_d{digits}_obs{args.num_obs}_lr{args.lr}_k{args.k1}"
    if args.k2 is not None:
        name += f"_v{args.k2}"
    out = next_run_dir(Path(args.path) / name)

    log = None
    if args.wandb:
        import wandb
        run = wandb.init(project=args.wandb, group=name, name=f"{name}_seed{args.seed}", config=vars(args))
        log = run.log

    X_train = mnist.load(args.side, split="train", digits=args.digits, max_samples=args.max_train)
    X_test = mnist.load(args.side, split="test", digits=args.digits, max_samples=5000, seed=args.seed)
    sigma = mnist.median_heuristic(X_train[:100])
    print(f"{len(X_train)} training images, digits {digits}, sigma = {sigma:.3f}")

    params0 = np.random.default_rng(args.seed).normal(size=model.num_params(args.side))
    params, losses, round_losses = train(
        args.side, sigma, X_train, params0,
        n_obs=args.num_obs, n_rounds=args.num_rounds, steps=args.num_steps,
        lr=args.lr, k1=args.k1, k2=args.k2, seed=args.seed, log=log,
    )

    mmd_mean, mmd_std = evaluate(
        args.side, sigma, X_test, params,
        n_obs=100, n_batches=10, k1=args.k1, k2=args.k2, seed=args.seed,
    )
    print(f"test batch MMD: {mmd_mean:.3e} ± {mmd_std:.3e}")
    if log:
        run.summary.update({"test/mmd_mean": mmd_mean, "test/mmd_std": mmd_std})
        run.finish()

    result = {
        "config": {**vars(args), "num_qubits": args.side ** 2, "sigma": sigma},
        "params": np.asarray(params).tolist(),
        "loss_history": losses,
        "round_loss_history": round_losses,
        "mmd_mean": mmd_mean,
        "mmd_std": mmd_std,
    }
    (out / "result.json").write_text(json.dumps(result, indent=2))
    save_qasm(lambda p: model.ansatz(p, args.side), params, out / "circuit.qasm")
    print(f"saved {out}/result.json and circuit.qasm")


if __name__ == "__main__":
    main()
