"""
A simple hardware-efficient ansatz for a 1D chain of L qubits.

The circuit is built from `depth` repeated blocks of:

    RY on every qubit  ->  CNOT entanglers on even bonds (0,1), (2,3), ...
    RX on every qubit  ->  CNOT entanglers on odd bonds  (1,2), (3,4), ...

followed by one final RY layer. Each rotation gate uses its own parameter,
so the total number of parameters is (2 * depth + 1) * L.

This is deliberately simple: a real hardware-efficient ansatz could use
more qubits per entangler, more rotation axes, or a different connectivity
pattern. The point here is to show the smallest circuit that is expressive
enough to approximate the ANNNI ground state, so the Pauli-propagation
machinery in run_vqe.py stays easy to follow.
"""
import pennylane as qml


def ansatz(params, L: int, depth: int = 2, show: bool = False) -> None:
    """
    Apply the ansatz's gates to the current circuit.

    params : a flat array of length (2 * depth + 1) * L.
    L      : number of qubits in the chain.
    depth  : number of RY/entangle/RX/entangle blocks (default 2).
    show   : if True, do not apply the gates. Instead, draw the circuit as
             ASCII art and print it to the terminal, which is useful to
             sanity-check the circuit's structure before running a full
             training. See the bottom of this file for a standalone example.
    """
    if show:
        _print_circuit(params, L, depth)
        return

    param_index = 0

    def rotation_layer(gate) -> None:
        nonlocal param_index
        for wire in range(L):
            gate(params[param_index], wires=wire)
            param_index += 1

    def entangling_layer(start_bond: int) -> None:
        # start_bond = 0 entangles bonds (0,1), (2,3), ... (the "even" bonds)
        # start_bond = 1 entangles bonds (1,2), (3,4), ... (the "odd" bonds)
        # Alternating even/odd bonds lets information spread across the
        # whole chain after just a couple of blocks, without ever needing a
        # gate between two qubits that are not nearest neighbors.
        for i in range(start_bond, L - 1, 2):
            qml.CNOT(wires=[i, i + 1])

    for _ in range(depth):
        rotation_layer(qml.RY)
        entangling_layer(start_bond=0)
        rotation_layer(qml.RX)
        entangling_layer(start_bond=1)

    # Final rotation layer, with no entangler after it.
    rotation_layer(qml.RY)


def num_params(L: int, depth: int = 2) -> int:
    """Number of trainable parameters the ansatz above expects."""
    return (2 * depth + 1) * L


def _print_circuit(params, L: int, depth: int) -> None:
    """
    Draw the ansatz as ASCII art using PennyLane's own drawer.

    This builds a throwaway QNode purely for drawing purposes: it is not
    used anywhere else in the example, and calling ansatz(..., show=True)
    only prints, it never applies gates to whatever circuit you were
    building.
    """
    device = qml.device("default.qubit", wires=L)

    @qml.qnode(device)
    def drawable_circuit(params):
        ansatz(params, L, depth=depth)
        return qml.expval(qml.PauliZ(0))

    print(qml.draw(drawable_circuit)(params))


if __name__ == "__main__":
    # Quick visual check: draw the circuit for a small chain with random
    # parameters, e.g.
    #     python ansatz.py --L 4 --depth 2
    import argparse

    import numpy as np

    parser = argparse.ArgumentParser(description="Draw the ANNNI example ansatz.")
    parser.add_argument("--L", type=int, default=4, help="Number of qubits.")
    parser.add_argument("--depth", type=int, default=2, help="Number of RY/CNOT/RX/CNOT blocks.")
    args = parser.parse_args()

    random_params = np.random.uniform(0, 2 * np.pi, num_params(args.L, args.depth))
    ansatz(random_params, args.L, depth=args.depth, show=True)
