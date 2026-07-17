"""
Training script for a quantum generative model on binarised MNIST images.

Overview
--------
The model is a parameterised quantum circuit (the shell ansatz in ansatz.py)
whose output distribution is matched to the empirical distribution of a
binarised, downsampled MNIST digit dataset.

The loss is a kernel MMD (Maximum Mean Discrepancy) proxy:

    L(θ) = Σ_k w_k (⟨O_k⟩_θ - μ_k)²

where each O_k is a tensor product of PauliZ operators, μ_k is the
corresponding empirical moment on the training set, and w_k is a weight
proportional to the multiplicity of O_k in the random batch.

Each training round:
  1. A fresh batch of random Pauli observables is drawn from a binomial
     distribution parameterised by the MMD bandwidth σ.
  2. A Propagator for that batch is built and analytically propagated.
  3. The batch MMD loss is minimised for ``steps_per_round`` steps - via
     ``pprop.optimization.adam`` (--device cpu) or the JAX-fused
     ``pprop.optimization.adam_gpu`` (--device gpu, ~8.5x faster at this
     script's defaults - see notebooks/test/gpu_backend.ipynb).

Usage
-----
    python train.py --side 8 --num_obs 20 --lr 0.05 --num_epochs 100 \\
                    --num_steps 20 --k1 6 --k2 20 --seed 0
    python train.py --digits 0 1 --side 8  # train on 0s and 1s only

    # --device: exactly two choices, deliberately (see notebooks/test/gpu_backend.ipynb
    # for the measurements behind both).
    #   'cpu' (default) -> backend='sparse' - the best CPU-only Propagator
    #                       evaluator measured in eval_and_grad_backends.ipynb.
    #   'gpu'           -> backend='vmap', device='gpu' - measured ~4-9x FASTER
    #                       than 'sparse' on CPU, for this script's typical
    #                       --num_obs (the crossover was ~4-16 observables per
    #                       call). Requires a CUDA-enabled jax/jaxlib. Opt-in
    #                       only (never automatic) - GPUs are a shared,
    #                       contended resource.
    python train.py --device cpu   # same as omitting --device
    python train.py --device gpu

    # --eval_n_jobs threads observable evaluation across CPU cores every Adam
    # step (backend='standard'/'sparse' only). Left at its default (1) below
    # deliberately: measured to make every step SLOWER at the term counts
    # k1/k2 truncation produces (Python's GIL dominates - each observable's
    # NumPy work is too small, ~0.1-0.2ms, to amortize thread overhead), at
    # every thread count tried from 2 to "all cores". Don't turn it on unless
    # you've re-measured for your own k1/k2/circuit and it actually helps.

    # Every run logs to Weights & Biases by default (--wandb_mode online),
    # so all jobs launched by TRAIN_cpu.sh and TRAIN_gpu.sh - regardless of
    # which node/partition they land on - show up together in one project's
    # dashboard, grouped by hyperparameters (see --wandb_project/_entity/
    # _mode and the wandb.init() call below). Use --wandb_mode offline on a
    # node without internet, or disabled to turn it off entirely.
"""

from __future__ import annotations

import argparse
import os
import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pennylane as qml
import wandb
from PIL import Image

from pprop import Propagator
from pprop.optimization import adam as pprop_adam, adam_gpu as pprop_adam_gpu
from pprop.propagator.pruning import DeadQubitPruner, XYWeightPruner

sys.path.append('./')
import ansatz

# ---------------------------------------------------------------------------
# MNIST dataset utilities
# ---------------------------------------------------------------------------

_MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}

_MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def download_mnist(data_dir: str | Path = "mnist_data") -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for key, fname in _MNIST_FILES.items():
        dest = data_dir / fname
        if not dest.exists():
            url = _MNIST_URLS[key]
            print(f"Downloading {fname} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            print("done")
    return data_dir


def _load_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"Expected magic 2051, got {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(n, rows, cols)


def _load_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"Expected magic 2049, got {magic}"
        return np.frombuffer(f.read(), dtype=np.uint8)


def load_mnist_raw(
    data_dir: str | Path = "mnist_data",
    split: str = "train",
    auto_download: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    if auto_download:
        download_mnist(data_dir)
    img_file = "train-images-idx3-ubyte.gz" if split == "train" else "t10k-images-idx3-ubyte.gz"
    lbl_file = "train-labels-idx1-ubyte.gz" if split == "train" else "t10k-labels-idx1-ubyte.gz"
    images = _load_idx_images(data_dir / img_file)
    labels = _load_idx_labels(data_dir / lbl_file)
    return images, labels


def downsample_images(images: np.ndarray, n_rows: int, n_cols: int) -> np.ndarray:
    out = np.empty((len(images), n_rows, n_cols), dtype=np.float32)
    for i, img in enumerate(images):
        if img.dtype != np.uint8:
            img = (img * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img, mode="L")
        pil_img = pil_img.resize((n_cols, n_rows), resample=Image.LANCZOS)
        arr = np.asarray(pil_img, dtype=np.float32) / 255.0
        out[i] = (arr > 0.3).astype(np.float32)
    return out


def get_mnist(
    n_rows: int,
    n_cols: int,
    *,
    data_dir: str | Path = "mnist_data",
    digits: list[int] | None = None,
    split: str = "train",
    max_samples: int | None = None,
    seed: int = 0,
) -> np.ndarray:
    raw_images, raw_labels = load_mnist_raw(data_dir, split)

    if digits is not None:
        mask = np.isin(raw_labels, digits)
        raw_images = raw_images[mask]
        raw_labels = raw_labels[mask]

    images = downsample_images(raw_images, n_rows, n_cols)

    if max_samples is not None and max_samples < len(images):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(images), max_samples, replace=False)
        images = images[idx]

    return images.reshape(len(images), n_rows * n_cols)


def median_heuristic(X: np.ndarray) -> float:
    n = len(X)
    dists = np.sqrt(np.sum((X[None] - X[:, None]) ** 2, axis=-1))
    return float(np.median(dists[np.triu_indices(n, k=1)]))


# ---------------------------------------------------------------------------
# Circuit and observable utilities
# ---------------------------------------------------------------------------

def circuit_maker(num_qubits: int, obs: list):
    def circuit(params):
        ansatz.ansatz(params, int(np.sqrt(num_qubits)))
        return [qml.expval(ob) for ob in obs]
    return circuit


def sample_observables_binomial(
    num_qubits: int,
    sigma: float,
    n_obs: int,
    rng: np.random.Generator,
) -> list[tuple[tuple[int, ...], object, int]]:
    t = np.tanh(1.0 / (4.0 * sigma ** 2))
    p = t / (1.0 + t)

    counts: dict[tuple[int, ...], int] = {}
    while len(counts) < n_obs:
        mask   = rng.random(num_qubits) < p
        qubits = tuple(int(q) for q in np.where(mask)[0])
        if len(qubits) == 0:
            continue
        counts[qubits] = counts.get(qubits, 0) + 1

    pool = []
    for qubits, count in counts.items():
        ob = qml.PauliZ(qubits[0])
        for q in qubits[1:]:
            ob = ob @ qml.PauliZ(q)
        pool.append((qubits, ob, count))
    return pool


def compute_data_moments(
    X: np.ndarray,
    pool: list[tuple[tuple[int, ...], object, int]],
) -> np.ndarray:
    spins = 1.0 - 2.0 * X
    moments = np.empty(len(pool))
    for k, (qubits, *_) in enumerate(pool):
        moments[k] = np.mean(np.prod(spins[:, qubits], axis=1))
    return moments


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_train_data(
    params: np.ndarray,
    num_qubits: int,
    loss_history: list[float],
    round_loss_history: list[float],
    test_mmd_mean: float,
    test_mmd_std: float,
    filename: str | Path,
) -> None:
    data = {
        "num_qubits":         num_qubits,
        "params":             np.array(params).tolist(),
        "loss_history":       loss_history,
        "round_loss_history": round_loss_history,
        "mmd_mean":           test_mmd_mean,
        "mmd_std":            test_mmd_std,
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    num_qubits: int,
    sigma: float,
    X_train: np.ndarray,
    params_init: np.ndarray,
    *,
    n_obs: int = 15,
    n_rounds: int = 40,
    steps_per_round: int = 20,
    lr: float = 0.05,
    k1: int | None = None,
    k2: int | None = None,
    seed: int = 0,
    backend: str = "sparse",
    eval_n_jobs: int = 1,
    device: str | None = None,
    wandb_run: wandb.sdk.wandb_run.Run | None = None,
) -> tuple[np.ndarray, list[float], list[float]]:
    rng       = np.random.default_rng(seed)
    params    = params_init.copy()
    all_losses: list[float] = []
    round_mean_losses: list[float] = []
    global_step = 0

    for r in range(n_rounds):
        print(f"{r}/{n_rounds}...")
        batch_pool = sample_observables_binomial(num_qubits, sigma, n_obs, rng)
        batch_mu   = compute_data_moments(X_train, batch_pool)
        batch_obs  = [ob      for _, ob, _     in batch_pool]
        weights    = np.array([c for _, _,  c  in batch_pool], dtype=float)
        weights   /= weights.sum()

        prop = Propagator(circuit_maker(num_qubits, batch_obs), k1=k1, k2=k2)
        prop.propagate(
            pruners=[XYWeightPruner(), DeadQubitPruner()],
            num_jobs=args.num_jobs,
            backend=backend,
            eval_n_jobs=eval_n_jobs,
            device=device,
        )

        mu = batch_mu
        w  = weights

        if backend == "vmap":
            # GPU-fused path (see pprop.optimization.adam_gpu): L must be
            # jax.numpy, not numpy, since it's traced into the same
            # jax.lax.scan-compiled program as the propagator's own JAX
            # computation. grad_L is left to jax.grad(L) - exact, and free
            # to get from a jnp loss (no separate closed-form needed).
            import jax.numpy as jnp
            mu_j, w_j = jnp.asarray(mu), jnp.asarray(w)

            def L(f_vals):
                return jnp.dot(w_j, (f_vals - mu_j) ** 2)

            result = pprop_adam_gpu(
                L, prop, np.array(params),
                lr          = lr,
                num_steps   = steps_per_round,
                print_every = 0,
            )
        else:
            def L(f_vals: np.ndarray) -> float:
                return float(np.dot(w, (f_vals - mu) ** 2))

            def grad_L(f_vals: np.ndarray) -> np.ndarray:
                return 2.0 * w * (f_vals - mu)

            result = pprop_adam(
                L, prop, np.array(params),
                lr          = lr,
                num_steps   = steps_per_round,
                print_every = 0,
                grad_L      = grad_L,
            )
        params = result["params"]
        round_losses = result["loss_history"]
        all_losses.extend(round_losses)
        global_step += steps_per_round

        mean_loss = float(np.mean(round_losses))
        round_mean_losses.append(mean_loss)
        print(f"  round {r+1:4d}/{n_rounds}  loss = {mean_loss:.8f}")

        if wandb_run is not None:
            # global_step (real Adam step count) is used as the x-axis
            # rather than round index r, so CPU runs (steps_per_round=1000)
            # and GPU runs (steps_per_round=4000) land on a comparable scale
            # in the dashboard.
            wandb_run.log({
                "round":            r + 1,
                "loss/round_mean":  mean_loss,
                "loss/round_min":   float(np.min(round_losses)),
                "loss/round_max":   float(np.max(round_losses)),
                "loss/round_last":  float(round_losses[-1]),
            }, step=global_step)

    return params, all_losses, round_mean_losses


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    num_qubits: int,
    params: np.ndarray,
    X_test: np.ndarray,
    *,
    n_obs: int = 100,
    n_samples: int = 10,
    k1: int = 6,
    k2: int = 20,
    sigma: float = 1.0,
    seed: int = 0,
    backend: str = "sparse",
    eval_n_jobs: int = 1,
    device: str | None = None,
) -> tuple[float, float]:
    rng_base  = np.random.default_rng(seed)
    mmd_values: list[float] = []

    for _ in range(n_samples):
        rng_val = np.random.default_rng(int(rng_base.integers(0, 2**31)))

        batch_pool = sample_observables_binomial(num_qubits, sigma, n_obs, rng_val)
        batch_mu   = compute_data_moments(X_test, batch_pool)
        batch_obs  = [ob      for _, ob, _     in batch_pool]
        weights    = np.array([c for _, _,  c  in batch_pool], dtype=float)
        weights   /= weights.sum()

        prop = Propagator(circuit_maker(num_qubits, batch_obs), k1=k1, k2=k2)
        prop.propagate(
            pruners=[XYWeightPruner(), DeadQubitPruner()],
            num_jobs=args.num_jobs,
            backend=backend,
            eval_n_jobs=eval_n_jobs,
            device=device,
        )

        residuals = prop(params) - batch_mu
        mmd_values.append(float(np.dot(weights, residuals ** 2)))

    return float(np.mean(mmd_values)), float(np.std(mmd_values))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train a quantum generative model on binarised MNIST."
    )
    parser.add_argument('--side',          type=int,   default=8,
                        help='Grid side length; num_qubits = side².')
    parser.add_argument('--num_obs',       type=int,   default=20,
                        help='Pauli observables sampled per training round.')
    parser.add_argument('--num_epochs',    type=int,   default=100,
                        help='Number of training rounds.')
    parser.add_argument('--lr',            type=float, default=0.1,
                        help='Adam learning rate.')
    parser.add_argument('--num_steps',     type=int,   default=5000,
                        help='Adam steps per round.')
    parser.add_argument('--k1',            type=int,   default=6,
                        help='Propagator k1 truncation parameter.')
    parser.add_argument('--k2',            type=int,   default=20,
                        help='Propagator k2 truncation parameter.')
    parser.add_argument('--seed',          type=int,   default=6,
                        help='Global random seed.')
    parser.add_argument('--path',          type=str,   default='./results',
                        help='Root output directory.')
    parser.add_argument('--data_dir',      type=str,   default='mnist_data',
                        help='Directory with (or for) MNIST IDX files.')
    parser.add_argument('--max_train',     type=int,   default=5000,
                        help='Maximum training samples.')
    parser.add_argument('--num_jobs',     type=int,   default=-1,
                        help="Worker processes used to propagate observables "
                             "in parallel (one-time cost per round). -1 "
                             "(default) uses every core actually allocated "
                             "to this process (via os.sched_getaffinity, so "
                             "this respects e.g. 'sbatch --cpus-per-task' "
                             "rather than the whole node's core count). "
                             "Avoid passing a number much larger than that: "
                             "each worker re-imports this whole module tree "
                             "(PennyLane, JAX, ...), so far more workers than "
                             "cores just adds startup overhead per round "
                             "with no extra parallelism to show for it.")
    parser.add_argument('--device',        type=str,   default='cpu',
                        choices=['cpu', 'gpu'],
                        help="Exactly two choices, deliberately (see "
                             "notebooks/test/gpu_backend.ipynb for the measurements "
                             "behind both, and eval_and_grad_backends.ipynb for "
                             "why 'standard' and plain 'vmap' aren't offered "
                             "here). 'cpu' (default) uses Propagator's "
                             "backend='sparse' - the best CPU-only evaluator "
                             "measured for this workload. 'gpu' uses "
                             "backend='vmap', device='gpu' - measured ~4-9x "
                             "faster than 'sparse' for this script's typical "
                             "--num_obs (crossover was ~4-16 observables per "
                             "call) - requires a CUDA-enabled jax/jaxlib. "
                             "Opt-in only, never automatic: GPUs on a shared "
                             "machine are a contended resource.")
    parser.add_argument('--eval_n_jobs',   type=int,   default=1,
                        help="Threads used to evaluate observables in "
                             "parallel per Adam step (backend='standard'/"
                             "'sparse' only). Default 1 (sequential) - "
                             "measured to make every step SLOWER at typical "
                             "k1/k2-truncated term counts, at every thread "
                             "count tried (Python's GIL dominates when each "
                             "observable's own NumPy work is this small). "
                             "-1 uses every core allocated to this process "
                             "(same convention as --num_jobs) if you want to "
                             "experiment, but re-measure before trusting it.")
    parser.add_argument('--digits',        type=int,   nargs='+',
                        default=list(range(10)),
                        metavar='D',
                        help='Digit classes to include (default: 0-9).')
    parser.add_argument('--wandb_project', type=str,   default='pauli-propagator-mnist',
                        help='W&B project name.')
    parser.add_argument('--wandb_entity',  type=str,   default=None,
                        help='W&B entity (team/user); default uses your '
                             'W&B account default.')
    parser.add_argument('--wandb_mode',    type=str,   default='online',
                        choices=['online', 'offline', 'disabled'],
                        help="'online' (default) streams live to the W&B "
                             "dashboard - this is what lets every job "
                             "launched by TRAIN_cpu.sh and TRAIN_gpu.sh show "
                             "up together in one place, grouped by "
                             "hyperparameters. 'offline' buffers locally for "
                             "a later `wandb sync` (e.g. a compute node "
                             "without internet). 'disabled' turns W&B off.")
    args = parser.parse_args()

    # --device is the only user-facing choice; translate it to the
    # Propagator-level backend/device pair internally. See --device's help
    # text for why only these two combinations are offered.
    backend = "sparse" if args.device == "cpu" else "vmap"
    device  = None     if args.device == "cpu" else "gpu"

    if backend == "vmap" and args.num_jobs != 1:
        # Propagator.propagate's docstring: num_jobs > 1 forks worker
        # processes AFTER JAX may have started background threads (which
        # happens as soon as backend="vmap" runs anything), and forking a
        # multithreaded process can deadlock the child. This isn't
        # hypothetical - it reproduces intermittently, not every run - so
        # don't leave it to chance: force num_jobs=1 whenever --device gpu
        # is selected, regardless of what --num_jobs was passed.
        print(f"--device gpu forces num_jobs=1 (was {args.num_jobs}) - "
              "forking worker processes after JAX starts threads risks a "
              "deadlock, see Propagator.propagate's docstring.")
        args.num_jobs = 1
    try:
        allocated_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        allocated_cpus = os.cpu_count()
    print(f"CPUs allocated to this process: {allocated_cpus}  "
          f"(--num_jobs={args.num_jobs}, --eval_n_jobs={args.eval_n_jobs}; "
          f"-1 means 'use all {allocated_cpus} of them')")
    print(f"--device {args.device}  ->  Propagator backend={backend!r}"
          + (f", device={device!r}" if device else ""))

    # Validate digit values
    invalid = [d for d in args.digits if not (0 <= d <= 9)]
    if invalid:
        parser.error(f"Invalid digit(s): {invalid}. Must be in 0-9.")

    num_qubits  = args.side * args.side
    digits_tag  = "".join(str(d) for d in sorted(args.digits))
    folder_name = (
        f"{args.side}_{args.num_obs}_{args.num_epochs}"
        f"_{args.lr}_{args.num_steps}_{args.k1}_{args.k2}"
        f"_d{digits_tag}"
    )
    out_path = Path(args.path) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    # group=<hyperparams, no seed> lets the dashboard fold every seed of a
    # given (side, num_obs, lr, k1, k2, digits, device) combo into one card;
    # name=<...+seed> keeps individual runs distinguishable within a group.
    # Both CPU and GPU sweeps share the same --path/folder_name convention,
    # so args.device is appended explicitly to avoid CPU/GPU runs of an
    # otherwise-identical hyperparameter grid colliding into the same group.
    wandb_group = f"{folder_name}_{args.device}"
    wandb_run = wandb.init(
        project = args.wandb_project,
        entity  = args.wandb_entity,
        mode    = args.wandb_mode,
        group   = wandb_group,
        name    = f"{wandb_group}_seed{args.seed}",
        tags    = [args.device, f"digits{digits_tag}"],
        config  = {**vars(args), "backend": backend, "num_qubits": num_qubits},
    )

    print(f"Training on digits: {sorted(args.digits)}")
    X_train = get_mnist(
        args.side, args.side,
        data_dir    = args.data_dir,
        digits      = args.digits,
        split       = "train",
        max_samples = args.max_train,
    )
    sigma = median_heuristic(X_train[:100])
    print(f"sigma = {sigma:.4f}")

    prop_dummy = Propagator(
        circuit_maker(num_qubits, [qml.PauliZ(0)]),
        k1=args.k1, k2=args.k2,
    )
    params_init = np.random.default_rng(args.seed).normal(
        loc=0.0, scale=1.0, size=prop_dummy.num_params
    )

    params_final, batch_losses, round_losses = train(
        num_qubits, sigma, X_train, params_init,
        n_obs           = args.num_obs,
        n_rounds        = args.num_epochs,
        steps_per_round = args.num_steps,
        lr              = args.lr,
        k1              = args.k1,
        k2              = args.k2,
        seed            = args.seed,
        backend         = backend,
        eval_n_jobs     = args.eval_n_jobs,
        device          = device,
        wandb_run       = wandb_run,
    )

    X_test = get_mnist(
        args.side, args.side,
        data_dir    = args.data_dir,
        digits      = args.digits,
        split       = "test",
        max_samples = 5000,
        seed        = args.seed,
    )
    test_mmd_mean, test_mmd_std = validate(
        num_qubits, params_final, X_test,
        n_obs=100, n_samples=10,
        k1=args.k1, k2=args.k2,
        sigma=sigma, seed=args.seed,
        backend=backend, eval_n_jobs=args.eval_n_jobs,
        device=device,
    )
    print(f"Test MMD: {test_mmd_mean:.6f} ± {test_mmd_std:.6f}")
    wandb_run.log({"test/mmd_mean": test_mmd_mean, "test/mmd_std": test_mmd_std})
    wandb_run.summary["test/mmd_mean"] = test_mmd_mean
    wandb_run.summary["test/mmd_std"]  = test_mmd_std

    idx = 0
    idx_path = out_path / f"{idx:03d}"
    while idx_path.exists():
        idx += 1
        idx_path = out_path / f"{idx:03d}"
    idx_path.mkdir()

    save_train_data(params_final, num_qubits, batch_losses, round_losses,
                    test_mmd_mean, test_mmd_std, idx_path / "result.json")
    print(f"Results saved to {idx_path / 'result.json'}")

    wandb_run.finish()
