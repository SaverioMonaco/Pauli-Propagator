"""
Binarised, downsampled MNIST for quantum generative modelling.

Each image is resized to ``side x side`` pixels and thresholded to {0, 1},
so one image is one bitstring on ``side**2`` qubits (pixel ``(r, c)`` lives
on qubit ``r * side + c``).

Conventions
-----------
A pixel value ``x`` maps to the PauliZ eigenvalue ``s = 1 - 2x``
(``x = 0 -> |0> -> +1``, ``x = 1 -> |1> -> -1``), so the empirical moment of
``Z_{q1} Z_{q2} ...`` on a dataset is the mean of ``s_{q1} s_{q2} ...``.

Example
-------
>>> from pprop import mnist
>>> X = mnist.load(side=8, digits=[0, 1], max_samples=1000)  # (1000, 64)
>>> mnist.z_moments(X, [(0,), (3, 4)])
"""

from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_URL = "https://storage.googleapis.com/cvdf-datasets/mnist/"
_FILES = {
    ("train", "images"): "train-images-idx3-ubyte.gz",
    ("train", "labels"): "train-labels-idx1-ubyte.gz",
    ("test",  "images"): "t10k-images-idx3-ubyte.gz",
    ("test",  "labels"): "t10k-labels-idx1-ubyte.gz",
}

#: Default download location; override with the ``PPROP_MNIST_DIR`` env var
#: or the ``data_dir`` argument of every loader.
DEFAULT_DATA_DIR = Path(
    os.environ.get("PPROP_MNIST_DIR", Path.home() / ".cache" / "pprop" / "mnist")
)


# ---------------------------------------------------------------------------
# Raw IDX files
# ---------------------------------------------------------------------------

def download(data_dir: str | Path = DEFAULT_DATA_DIR) -> Path:
    """Download the four MNIST IDX files into ``data_dir`` (skips existing ones)."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname in _FILES.values():
        dest = data_dir / fname
        if not dest.exists():
            print(f"Downloading {fname} ...", end=" ", flush=True)
            urllib.request.urlretrieve(_URL + fname, dest)
            print("done")
    return data_dir


def _read_idx(path: Path, magic_expected: int, header_ints: int) -> tuple[tuple[int, ...], np.ndarray]:
    with gzip.open(path, "rb") as f:
        magic, *shape = struct.unpack(">" + "I" * header_ints, f.read(4 * header_ints))
        if magic != magic_expected:
            raise ValueError(f"{path}: expected magic {magic_expected}, got {magic}")
        return tuple(shape), np.frombuffer(f.read(), dtype=np.uint8)


def load_raw(
    split: str = "train",
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the original 28x28 ``uint8`` images and their labels.

    Parameters
    ----------
    split : {"train", "test"}
    data_dir : path
        Where the IDX files live; they are downloaded there if missing.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    data_dir = download(data_dir)
    (n, rows, cols), images = _read_idx(data_dir / _FILES[split, "images"], 2051, 4)
    _, labels = _read_idx(data_dir / _FILES[split, "labels"], 2049, 2)
    return images.reshape(n, rows, cols), labels


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def binarize(images: np.ndarray, side: int, threshold: float = 0.3) -> np.ndarray:
    """Resize ``uint8`` images to ``side x side`` (Lanczos) and threshold to {0, 1}.

    Returns a ``float32`` array of shape ``(n, side, side)``.
    """
    from PIL import Image  # optional dependency, only needed here

    out = np.empty((len(images), side, side), dtype=np.float32)
    for i, img in enumerate(images):
        small = Image.fromarray(img, mode="L").resize((side, side), resample=Image.LANCZOS)
        out[i] = np.asarray(small, dtype=np.float32) / 255.0 > threshold
    return out


def load(
    side: int,
    *,
    split: str = "train",
    digits: Iterable[int] | None = None,
    max_samples: int | None = None,
    threshold: float = 0.3,
    seed: int = 0,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> np.ndarray:
    """Load binarised MNIST as flat bitstrings.

    Parameters
    ----------
    side : int
        Images are resized to ``side x side``; one image = ``side**2`` qubits.
    split : {"train", "test"}
    digits : iterable of int, optional
        Keep only these classes (default: all ten).
    max_samples : int, optional
        Random subset of this size (drawn with ``seed``).
    threshold : float
        Grey level in [0, 1] above which a pixel is set to 1.

    Returns
    -------
    ndarray of shape ``(n, side**2)`` with values in {0, 1}.
    """
    images, labels = load_raw(split, data_dir)
    if digits is not None:
        images = images[np.isin(labels, list(digits))]
    if max_samples is not None and max_samples < len(images):
        idx = np.random.default_rng(seed).choice(len(images), max_samples, replace=False)
        images = images[np.sort(idx)]
    return binarize(images, side, threshold).reshape(len(images), side * side)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def to_spins(X: np.ndarray) -> np.ndarray:
    """Map bits {0, 1} to PauliZ eigenvalues {+1, -1}."""
    return 1.0 - 2.0 * np.asarray(X, dtype=float)


def z_moments(X: np.ndarray, subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """Empirical ``<Z_{q1} Z_{q2} ...>`` for each qubit subset, over the rows of ``X``."""
    spins = to_spins(X)
    return np.array([spins[:, list(qubits)].prod(axis=1).mean() for qubits in subsets])


def median_heuristic(X: np.ndarray) -> float:
    """Median pairwise Euclidean distance; the usual Gaussian-kernel bandwidth."""
    X = np.asarray(X, dtype=float)
    d = np.sqrt(((X[:, None] - X[None]) ** 2).sum(-1))
    return float(np.median(d[np.triu_indices(len(X), k=1)]))


def mmd(X: np.ndarray, Y: np.ndarray, sigma: float) -> float:
    """Exact (biased, V-statistic) Gaussian-kernel MMD² between two sample sets.

    ``k(x, y) = exp(-|x - y|² / (2 sigma²))``. Both sets are bitstring arrays
    of shape ``(n, num_qubits)``; this is the quantity the stochastic Pauli-Z
    loss in ``mnistgen/train.py`` estimates.
    """
    def k(A, B):
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.exp(-d2 / (2 * sigma ** 2))

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    return float(k(X, X).mean() + k(Y, Y).mean() - 2 * k(X, Y).mean())


# ---------------------------------------------------------------------------
# Hardware samples <-> images
# ---------------------------------------------------------------------------

def bitstrings_to_array(bitstrings: Sequence[str], little_endian: bool = True) -> np.ndarray:
    """Turn measured bitstrings into an ``(n, num_qubits)`` {0, 1} array.

    Qiskit prints qubit 0 as the *rightmost* character (``little_endian=True``);
    pass ``False`` for strings where qubit 0 comes first.
    """
    arr = np.array([[int(b) for b in s] for s in bitstrings], dtype=np.float32)
    return arr[:, ::-1] if little_endian else arr


def as_images(X: np.ndarray) -> np.ndarray:
    """Reshape flat ``(n, side**2)`` bitstrings to ``(n, side, side)`` images."""
    side = int(round(np.sqrt(X.shape[1])))
    return np.asarray(X).reshape(len(X), side, side)
