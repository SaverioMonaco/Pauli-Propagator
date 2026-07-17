"""
Minimal VQE for one point of the ANNNI model, using Pauli propagation
instead of a shot-based or backprop-through-simulator quantum circuit
evaluation. This is the core loop behind the paper's results, stripped down
to a single (kappa, h) point: no DMRG comparison, no warm-starting across a
grid, no retries. See scripts/annni/train.py in the repository root for the
full version used to produce the paper's data.

What "Pauli propagation" buys us here: normally, evaluating <psi(theta)|O|psi(theta)>
and its gradient requires either running the circuit on a simulator/device
for every parameter update, or symbolic differentiation through a
simulator. Instead, the Propagator class evolves each observable backward
through the circuit's gates once (the Heisenberg picture), producing a
closed-form trigonometric expression in the parameters theta. After that
one-time propagation, evaluating the expectation value and its exact
gradient at any theta is just plugging numbers into that expression, which
is very cheap and requires no simulator at all.

Usage
-----
    python run_vqe.py --L 8 --kappa 0.5 --h 1.0 --steps 300

Try a few different --h values at fixed --kappa to see the ground-state
energy change as the transverse field crosses the model's phase boundaries.
"""
import argparse
import time

import numpy as np
import pennylane as qml

from ansatz import ansatz, num_params
from hamiltonian import full_hamiltonian, hamiltonian_observables, term_counts
from pprop import Propagator
from pprop.optimization import adam
from pprop.propagator.pruning import DeadQubitPruner, XYWeightPruner


def make_loss(L: int, J: float, kappa: float, h: float):
    """
    Build the scalar loss L(f) and its exact gradient dL/df, where f is the
    vector of *bare* observable expectation values returned by the
    propagated circuit (see hamiltonian_observables()).

    Because the Hamiltonian is just a fixed linear combination of those
    bare observables, L(f) is linear in f and its gradient dL/df is a
    constant vector, so we can write it down directly instead of relying on
    (slower, approximate) finite differences.
    """
    n_magnetic, n_nearest, n_axial = term_counts(L)

    def loss(f: np.ndarray) -> float:
        magnetic = f[:n_magnetic].sum()
        nearest = f[n_magnetic:n_magnetic + n_nearest].sum()
        axial = f[n_magnetic + n_nearest:].sum()
        return float(-J * h * magnetic - J * nearest + J * kappa * axial)

    def grad_loss(f: np.ndarray) -> np.ndarray:
        grad = np.empty(len(f))
        grad[:n_magnetic] = -J * h
        grad[n_magnetic:n_magnetic + n_nearest] = -J
        grad[n_magnetic + n_nearest:] = J * kappa
        return grad

    return loss, grad_loss


def exact_ground_state_energy(L: int, J: float, kappa: float, h: float) -> float:
    """
    Diagonalize the full Hamiltonian matrix exactly and return its lowest
    eigenvalue, i.e. the true ground-state energy for this (kappa, h)
    point.

    This is only meant as a sanity check for small L (the matrix has size
    2**L x 2**L, so this becomes slow well before L=20). The paper compares
    against DMRG for the larger chains it actually studies; exact
    diagonalization here is just the simplest ground truth for a small
    tutorial example like this one.
    """
    H = full_hamiltonian(L, J, kappa, h)
    H_matrix = qml.matrix(H, wire_order=range(L))
    eigenvalues = np.linalg.eigvalsh(H_matrix)
    return float(eigenvalues[0])


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    # 1. Build the circuit: our ansatz, followed by the expectation value of
    #    every bare observable making up the Hamiltonian. Propagator only
    #    needs this function, it will call it once itself to record the
    #    gate sequence.
    observables = hamiltonian_observables(args.L)

    def circuit(params):
        ansatz(params, args.L, depth=args.depth)
        return [qml.expval(o) for o in observables]

    # 2. Wrap the circuit in a Propagator. k1/k2 bound how large the
    #    Pauli-propagated expressions are allowed to grow: k1 discards
    #    terms whose Pauli weight (number of non-identity sites) exceeds
    #    k1, k2 discards terms whose combined trigonometric frequency
    #    exceeds k2. Both are optional (pass None to disable), and both
    #    trade a small, controllable amount of accuracy for a much smaller,
    #    faster expression as L grows.
    prop = Propagator(circuit, k1=args.k1, k2=args.k2)
    print(prop)

    # 3. Propagate every observable backward through the circuit once. This
    #    is the expensive, one-time step; XYWeightPruner and DeadQubitPruner
    #    are extra heuristics that discard terms unlikely to matter, on top
    #    of the k1/k2 cutoffs above.
    print("Propagating observables through the circuit (one-time cost)...")
    t0 = time.time()
    prop.propagate(pruners=[XYWeightPruner(), DeadQubitPruner()])
    print(f"  done in {time.time() - t0:.2f}s")

    # 4. Combine the propagated bare observables into the scalar energy
    #    loss for this specific (J, kappa, h) point, with an exact
    #    gradient, so Adam does not need to fall back on finite
    #    differences.
    loss, grad_loss = make_loss(args.L, args.J, args.kappa, args.h)

    # 5. Train with Adam. Every step calls prop.eval_and_grad(params), which
    #    evaluates the already-propagated expressions (and their exact
    #    parameter gradients) at the current params, no simulator involved.
    params_init = 2 * np.pi * (rng.random(num_params(args.L, args.depth)) - 0.5)
    print(f"Training {args.steps} Adam steps "
          f"(L={args.L}, J={args.J}, kappa={args.kappa}, h={args.h}, lr={args.lr})...")
    t0 = time.time()
    result = adam(
        L=loss,
        grad_L=grad_loss,
        propagator=prop,
        params_init=params_init,
        lr=args.lr,
        num_steps=args.steps,
        print_every=max(1, args.steps // 10),
    )
    print(f"  done in {time.time() - t0:.2f}s")

    vqe_energy = result["fun"]
    print(f"\nVQE energy (Pauli-propagated): {vqe_energy:.8f}")

    # 6. Optional sanity check: for small L, diagonalize the Hamiltonian
    #    exactly and compare. The VQE energy should sit close to (but,
    #    since the ansatz and optimizer are not perfect, not necessarily
    #    exactly at) this value.
    if args.check_exact:
        if args.L > 14:
            print("Skipping exact diagonalization: --L is too large "
                  "(2**L grows fast, try --L <= 14).")
        else:
            exact_energy = exact_ground_state_energy(args.L, args.J, args.kappa, args.h)
            print(f"Exact ground-state energy:    {exact_energy:.8f}")
            print(f"Absolute error:                {abs(vqe_energy - exact_energy):.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Minimal VQE for a single ANNNI (kappa, h) point via Pauli propagation."
    )
    parser.add_argument("--L", type=int, default=8,
                        help="Chain length (number of qubits). Kept small by "
                             "default so --check_exact stays fast.")
    parser.add_argument("--depth", type=int, default=2,
                        help="Number of blocks in the ansatz (see ansatz.py).")
    parser.add_argument("--J", type=float, default=1.0,
                        help="Overall energy scale.")
    parser.add_argument("--kappa", type=float, default=0.5,
                        help="Axial (next-nearest-neighbor) coupling ratio.")
    parser.add_argument("--h", type=float, default=1.0,
                        help="Transverse field strength.")
    parser.add_argument("--k1", type=int, default=6,
                        help="Propagator Pauli-weight cutoff (None to disable).")
    parser.add_argument("--k2", type=int, default=16,
                        help="Propagator frequency cutoff (None to disable).")
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Adam learning rate.")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of Adam optimization steps.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for the initial parameters.")
    parser.add_argument("--check_exact", action="store_true",
                        help="Also compute the exact ground-state energy via "
                             "full diagonalization, and compare against the "
                             "VQE result. Only practical for small --L.")
    args = parser.parse_args()

    main(args)
