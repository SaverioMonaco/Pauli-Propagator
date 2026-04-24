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
  3. ``pprop.optimization.adam`` minimises the batch MMD loss for
     ``steps_per_round`` steps.

Usage
-----
    python train.py --side 8 --num_obs 20 --lr 0.05 --num_epochs 100 \\
                    --num_steps 20 --k1 6 --k2 20 --seed 0
"""

from __future__ import annotations

import argparse
import gzip
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pennylane as qml
from PIL import Image

from pprop import Propagator
from pprop.optimization import adam as pprop_adam
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
    test_mmd_mean: float,
    test_mmd_std: float,
    filename: str | Path,
) -> None:
    data = {
        "num_qubits":    num_qubits,
        "params":        np.array(params).tolist(),
        "loss_history":  loss_history,
        "mmd_mean":      test_mmd_mean,
        "mmd_std":       test_mmd_std,
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
) -> tuple[np.ndarray, list[float]]:
    rng       = np.random.default_rng(seed)
    params    = params_init.copy()
    all_losses: list[float] = []
    global_step = 0

    for r in range(n_rounds):
        batch_pool = sample_observables_binomial(num_qubits, sigma, n_obs, rng)
        batch_mu   = compute_data_moments(X_train, batch_pool)
        batch_obs  = [ob      for _, ob, _     in batch_pool]
        weights    = np.array([c for _, _,  c  in batch_pool], dtype=float)
        weights   /= weights.sum()

        prop = Propagator(circuit_maker(num_qubits, batch_obs), k1=k1, k2=k2)
        prop.propagate(pruners=[XYWeightPruner(), DeadQubitPruner()])

        mu = batch_mu
        w  = weights

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
        print(f"  round {r+1:4d}/{n_rounds}  loss = {mean_loss:.8f}")

    return params, all_losses


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
        prop.propagate(pruners=[XYWeightPruner(), DeadQubitPruner()])

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
    parser.add_argument('--num_steps',     type=int,   default=10000,
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
    args = parser.parse_args()

    num_qubits  = args.side * args.side
    folder_name = (
        f"{args.side}_{args.num_obs}_{args.num_epochs}"
        f"_{args.lr}_{args.num_steps}_{args.k1}_{args.k2}"
    )
    out_path = Path(args.path) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    X_train = get_mnist(
        args.side, args.side,
        data_dir    = args.data_dir,
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

    params_final, batch_losses = train(
        num_qubits, sigma, X_train, params_init,
        n_obs           = args.num_obs,
        n_rounds        = args.num_epochs,
        steps_per_round = args.num_steps,
        lr              = args.lr,
        k1              = args.k1,
        k2              = args.k2,
        seed            = args.seed,
    )

    X_test = get_mnist(
        args.side, args.side,
        data_dir    = args.data_dir,
        split       = "test",
        max_samples = 5000,
        seed        = args.seed,
    )
    test_mmd_mean, test_mmd_std = validate(
        num_qubits, params_final, X_test,
        n_obs=100, n_samples=10,
        k1=args.k1, k2=args.k2,
        sigma=sigma, seed=args.seed,
    )
    print(f"Test MMD: {test_mmd_mean:.6f} ± {test_mmd_std:.6f}")

    idx = 0
    idx_path = out_path / f"{idx:03d}"
    while idx_path.exists():
        idx += 1
        idx_path = out_path / f"{idx:03d}"
    idx_path.mkdir()

    save_train_data(params_final, num_qubits, batch_losses, test_mmd_mean, test_mmd_std,
                    idx_path / "result.json")
    print(f"Results saved to {idx_path / 'result.json'}")
