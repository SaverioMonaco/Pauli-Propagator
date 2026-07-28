"""
VQE training script for the 2D transverse-field Ising model.

Builds a Propagator over the ansatz circuit, propagates it once, then
runs Adam to minimise ⟨ψ(θ)|H|ψ(θ)⟩.  Results are written to
``<path>/side<side>_k<k1>_v<k2>/result.json``.

Usage
-----
    python train.py --side 8 --J 1.0 --h 1.0 --k1 64 --k2 256 \\
                    --lr 5e-3 --num_steps 1000 --seed 0 --path .
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pennylane as qml

from pprop import Propagator
from pprop.optimization import adam

sys.path.append('./')
import ansatz
import ising2d


def main(
    side: int,
    J: float,
    h: float,
    k1: int,
    k2: int,
    lr: float,
    num_steps: int,
    seed: int,
    path: str,
) -> None:
    num_qubits = side * side

    def circuit(params):
        ansatz.ansatz(params, side)
        return qml.expval(ising2d.hamiltonian(side, J, h))

    print("VQE:")

    prop = Propagator(circuit, k1=k1, k2=k2)
    print(prop)

    print(f" Pauli Weight Cutoff: {k1}")
    print(f" Frequency Cutoff:    {k2}")
    print("Propagating...")

    time_start = time.time()
    prop.propagate(use_xy_weight_pruner=True, use_dead_qubit_pruner=True)
    time_end = time.time()
    print(f"  Time: {time_end - time_start:.3f} seconds")

    rng = np.random.default_rng(seed)
    params_init = 2 * np.pi * (rng.random(prop.num_params) - 0.5)

    print("Optimising...")
    result = adam(
        L=lambda f: f[0],
        propagator=prop,
        params_init=params_init,
        lr=lr,
        num_steps=num_steps,
    )

    result["num_qubits"] = num_qubits
    result["J"] = J
    result["h"] = h

    folder_name = f"side{side}_k{k1}_v{k2}"
    out_path = Path(path) / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    with open(out_path / "result.json", "w") as f:
        json.dump(result, f)

    print(f"Results saved to {(out_path / 'result.json').absolute()}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="VQE training for the 2D transverse-field Ising model."
    )
    parser.add_argument('--side',      type=int,   default=8,
                        help='Grid side length; num_qubits = side².')
    parser.add_argument('--J',         type=float, default=1.0,
                        help='Ising coupling constant.')
    parser.add_argument('--h',         type=float, default=1.0,
                        help='Transverse field strength.')
    parser.add_argument('--k1',        type=int,   default=64,
                        help='Propagator Pauli-weight cutoff.')
    parser.add_argument('--k2',        type=int,   default=256,
                        help='Propagator frequency cutoff.')
    parser.add_argument('--lr',        type=float, default=5e-3,
                        help='Adam learning rate.')
    parser.add_argument('--num_steps', type=int,   default=1000,
                        help='Number of optimisation steps.')
    parser.add_argument('--seed',      type=int,   default=0,
                        help='Random seed for parameter initialisation.')
    parser.add_argument('--path',      type=str,   default='.',
                        help='Root output directory.')
    args = parser.parse_args()

    main(
        side      = args.side,
        J         = args.J,
        h         = args.h,
        k1        = args.k1,
        k2        = args.k2,
        lr        = args.lr,
        num_steps = args.num_steps,
        seed      = args.seed,
        path      = args.path,
    )
