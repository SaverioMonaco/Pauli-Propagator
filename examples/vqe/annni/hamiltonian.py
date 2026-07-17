"""
The ANNNI (Axial Next-Nearest-Neighbor Ising) Hamiltonian, open boundary
conditions only, for a chain of L qubits:

    H(J, kappa, h) = -J * sum_i ( X_i X_{i+1}  -  kappa * X_i X_{i+2}  +  h * Z_i )

J sets the overall energy scale, kappa is the strength of the
next-nearest-neighbor coupling relative to the nearest-neighbor one, and h
is the transverse field. This model has a rich phase diagram in the
(kappa, h) plane (paramagnetic, ferromagnetic and antiphase regions), which
is why the paper studies it: it is simple to write down but hard for a
classical computer to solve everywhere, making it a good benchmark for a
variational quantum eigensolver (VQE).
"""
import pennylane as qml


def hamiltonian_observables(L: int) -> list:
    """
    Return the Hamiltonian's *bare* Pauli observables, i.e. without their
    J/kappa/h coefficients attached.

    We keep the observables and their coefficients separate on purpose.
    Pauli propagation (see ansatz.py and run_vqe.py) evolves each observable
    through the circuit once, and that work does not depend on J, kappa or
    h at all, only on L. So if you wanted to scan many (kappa, h) points for
    the same L, you would call this function once and reuse the result,
    instead of rebuilding a new Hamiltonian (and re-propagating it) for
    every point. run_vqe.py only trains a single point, but is written the
    same way so it is easy to extend into a scan.

    The observables come back in three groups, in this order:
      1. L single-qubit Z_i terms      (the magnetic / transverse-field term)
      2. L-1 two-qubit X_i X_{i+1}     (nearest-neighbor interaction)
      3. L-2 two-qubit X_i X_{i+2}     (axial next-nearest-neighbor interaction)

    term_counts() below returns the length of each group, so the flat list
    returned here can be split back into the three groups without
    re-deriving which indices belong to which term.
    """
    observables = []

    # 1. Magnetic term: one Z per site.
    for i in range(L):
        observables.append(qml.PauliZ(i))

    # 2. Nearest-neighbor interaction: X_i X_{i+1} for every bond along the chain.
    for i in range(L - 1):
        observables.append(qml.PauliX(i) @ qml.PauliX(i + 1))

    # 3. Axial next-nearest-neighbor interaction: X_i X_{i+2}.
    for i in range(L - 2):
        observables.append(qml.PauliX(i) @ qml.PauliX(i + 2))

    return observables


def term_counts(L: int) -> tuple[int, int, int]:
    """
    Number of (magnetic, nearest-neighbor, axial) terms for a chain of L
    qubits with open boundaries, matching the group order and sizes
    returned by hamiltonian_observables() above.
    """
    n_magnetic = L
    n_nearest = L - 1
    n_axial = L - 2
    return n_magnetic, n_nearest, n_axial


def full_hamiltonian(L: int, J: float, kappa: float, h: float) -> qml.Hamiltonian:
    """
    Build the full Hamiltonian, coefficients included, as a single
    qml.Hamiltonian object.

    This is only used for the "exact ground state" sanity check in
    run_vqe.py (via exact diagonalization). The VQE training loop itself
    never builds this object: it works directly with the bare observables
    above and their coefficients, combined by hand into a scalar loss (see
    make_loss() in run_vqe.py), which is cheaper and is how the paper's
    experiments actually run.
    """
    observables = hamiltonian_observables(L)
    n_magnetic, n_nearest, n_axial = term_counts(L)

    coefficients = (
        [-J * h] * n_magnetic
        + [-J] * n_nearest
        + [J * kappa] * n_axial
    )

    return qml.Hamiltonian(coefficients, observables)
