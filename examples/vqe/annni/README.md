# VQE on the ANNNI model, via Pauli propagation

A minimal, heavily commented walkthrough of how this repository's paper
results were produced: a variational quantum eigensolver (VQE) for the
ANNNI spin chain, trained without ever running a quantum circuit simulator
during the optimization loop.

This example is meant to be read top to bottom, not just run. If you only
want the full-featured version used to generate the paper's data (grid
sweeps, DMRG comparison, warm-started rows, retries), see
`scripts/annni/train.py` in the repository root instead. Everything here is
a stripped-down, single-point version of the same idea.

## The model

The ANNNI (Axial Next-Nearest-Neighbor Ising) Hamiltonian on a chain of `L`
qubits with open boundaries is

$$
H(J, \kappa, h) = -J \sum_i \left( X_i X_{i+1} - \kappa X_i X_{i+2} + h Z_i \right)
$$

$J$ is an overall energy scale, $\kappa$ is the strength of the axial
(next-nearest-neighbor) coupling relative to the nearest-neighbor one, and
$h$ is a transverse field. Depending on $\kappa$ and $h$, the ground state of
this model sits in different phases (paramagnetic, ferromagnetic,
antiphase), which makes it a good, non-trivial benchmark for a VQE.

## Why Pauli propagation

A normal VQE loop evaluates $\langle \psi(\theta) | H | \psi(\theta) \rangle$ and its gradient
by running the circuit on a simulator (or real device) at each optimization
step. Here, we instead evolve each observable in $H$ backward through the
ansatz's gates once, in the Heisenberg picture. That single pass produces a
closed-form trigonometric expression in the circuit's parameters $\theta$.
After that, evaluating the expectation value (and its exact gradient) at
any $\theta$ is just plugging numbers into that expression: no simulator
calls, no shots, no backpropagation through a circuit.

This is what `pprop.Propagator` does, and it is the only reason this
example can afford to run hundreds of Adam steps quickly even without a
GPU.

## Files

- `hamiltonian.py`: the ANNNI Hamiltonian, split into its bare observables
  (`Z_i`, `X_i X_{i+1}`, `X_i X_{i+2}`) so they can be propagated once and
  reused across different `(J, kappa, h)` combinations.
- `ansatz.py`: a small hardware-efficient ansatz (alternating RY/RX
  rotation layers and CNOT entanglers). Also runnable on its own to print
  an ASCII drawing of the circuit, e.g. `python ansatz.py --L 4 --depth 2`.
- `run_vqe.py`: builds the circuit, propagates the Hamiltonian's
  observables through it once, then trains the ansatz's parameters with
  Adam to minimize the energy at a chosen `(kappa, h)` point.

## Running it

From this directory (make sure `pprop` is installed first, see the
repository root `README.md`, e.g. `pip install -e .` from there):

```bash
python run_vqe.py --L 8 --kappa 0.5 --h 1.0 --steps 300
```

This prints the propagated circuit's size, the one-time propagation time,
the Adam loss every few steps, and the final VQE energy.

To also check the result against the true ground-state energy (exact
diagonalization of the full Hamiltonian matrix, only practical for small
`L`, say `L <= 14`):

```bash
python run_vqe.py --L 6 --kappa 0.5 --h 1.0 --steps 2000 --lr 0.01 --check_exact
```

```
VQE energy (Pauli-propagated): -6.98889377
Exact ground-state energy:    -7.00737323
Absolute error:                1.848e-02
```

Try sweeping `--h` at a fixed `--kappa` to see the ground-state energy
change as the model crosses a phase boundary; that is exactly the kind of
scan `scripts/annni/train.py` automates for the paper's actual results,
warm-starting each point from the previous one's trained parameters.

## Parameters worth experimenting with

- `--k1` / `--k2`: cutoffs the Propagator uses to discard Pauli terms that
  grow too large (`k1`, by Pauli weight) or too high-frequency (`k2`) during
  propagation. Lower values are faster but less accurate; raise them if the
  VQE energy is far from the exact one at small `L`, or lower them to see
  how much accuracy you can trade away for speed at larger `L`.
- `--depth`: number of rotation/entangling blocks in the ansatz. A deeper
  ansatz is more expressive but also gives the Propagator more gates to
  push observables through, so propagation gets slower too.
