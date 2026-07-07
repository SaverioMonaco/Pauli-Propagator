"""Deploy a trained VQE circuit to IBM Quantum hardware.

Loads a trained parameter vector from a ``result.json`` file, converts the
PennyLane circuit to OpenQASM 2, transpiles it for the target backend, and
submits Estimator jobs for 10 evenly-spaced checkpoints from the optimisation
history.  Results are saved alongside the input file as ``deploy.json``.

Usage
-----
    python deploy.py --path /path/to/side8_k64_v256 --backend ibm_berlin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from qiskit import qasm2
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService

sys.path.append('./')
import ansatz
from ising2d import hamiltonian


def pennylane_ham_to_qiskit(pl_ham: qml.Hamiltonian, num_qubits: int) -> SparsePauliOp:
    pauli_map = {"PauliX": "X", "PauliY": "Y", "PauliZ": "Z", "Identity": "I"}
    pauli_list = []
    for coeff, op in zip(pl_ham.coeffs, pl_ham.ops):
        pauli_str = ["I"] * num_qubits
        factors = op.operands if hasattr(op, "operands") else [op]
        for factor in factors:
            wire = factor.wires[0]
            pauli_str[wire] = pauli_map[type(factor).__name__]
        pauli_list.append(("".join(reversed(pauli_str)), coeff))
    return SparsePauliOp.from_list(pauli_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy a trained VQE circuit to IBM Quantum hardware."
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the result directory containing result.json.",
    )
    parser.add_argument(
        "--backend", type=str, default="ibm_berlin",
        help="IBM backend name (default: ibm_berlin).",
    )
    parser.add_argument(
        "--channel", type=str, default="ibm_cloud",
        help="IBM Runtime channel (default: ibm_cloud).",
    )
    args = parser.parse_args()

    path = Path(args.path)

    # ------------------------------------------------------------------ #
    # Load trained parameters.
    # ------------------------------------------------------------------ #
    with open(path / "result.json") as f:
        data = json.load(f)
    params_opt     = data["params"]
    params_history = data["params_history"]
    num_qubits     = data["num_qubits"]
    J              = data.get("J", 1)
    h              = data.get("h", 1)
    side           = int(np.sqrt(num_qubits))

    n       = len(params_history)
    indices = np.linspace(0, n - 1, 10, dtype=int)[5:]
    print(indices)
    reduced_params_history = [params_history[i] for i in indices]

    print("deploy.py")
    print(f"  path:       {path}")
    print(f"  num_params: {len(params_opt)}")

    # ------------------------------------------------------------------ #
    # Build Hamiltonian in Qiskit form.
    # ------------------------------------------------------------------ #
    H_pl     = hamiltonian(side=side, J=J, h=h)
    H_qiskit = pennylane_ham_to_qiskit(H_pl, num_qubits)

    backend = QiskitRuntimeService(channel=args.channel).backend(args.backend)
    print(f"  backend:    {backend.name}")
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)

    # ------------------------------------------------------------------ #
    # Run Estimator for each checkpoint.
    # ------------------------------------------------------------------ #
    epochs_list     = []
    params_list     = []
    expval_opt_list = []
    std_opt_list    = []

    for i, epoch in enumerate(indices):
        params = params_opt if i == len(indices) - 1 else reduced_params_history[i]

        def build_isa_circuit(p):
            dev = qml.device("default.qubit", wires=num_qubits)

            @qml.qnode(dev)
            def pennylane_circuit():
                ansatz.ansatz(p, side)
                return qml.state()

            openqasm_circuit = qml.to_openqasm(pennylane_circuit)
            qiskit_circuit   = qasm2.loads(openqasm_circuit())
            return pm.run(qiskit_circuit)

        print(f"({i+1}/{len(indices)}) Building ISA circuit...")
        isa_opt   = build_isa_circuit(params)
        H_isa_opt = H_qiskit.apply_layout(isa_opt.layout)

        print("Running Estimator...")
        estimator  = Estimator(backend)
        job        = estimator.run([(isa_opt, H_isa_opt)])
        results    = job.result()
        pub_result = results[0]

        print("Shots used:", pub_result.metadata.get("shots", "N/A"))
        print("Std dev:",    pub_result.data.stds)

        expval_opt = float(results[0].data.evs)
        std_opt    = float(results[0].data.stds)
        print(f"<H> optimized = {expval_opt:.6f}")

        epochs_list.append(int(epoch))
        params_list.append(params)
        expval_opt_list.append(expval_opt)
        std_opt_list.append(std_opt)

    two_qubit_depth = isa_opt.depth(
        filter_function=lambda inst: inst.operation.num_qubits == 2
    )

    # ------------------------------------------------------------------ #
    # Save results.
    # ------------------------------------------------------------------ #
    output = {
        "path":          str(path),
        "backend":       backend.name,
        "num_qubits":    num_qubits,
        "num_shots":     pub_result.metadata.get("shots", "N/A"),
        "J":             J,
        "h":             h,
        "side":          side,
        "circuit_depth": two_qubit_depth,
        "optimized": {
            "epochs": epochs_list,
            "params": params_list,
            "expval": expval_opt_list,
            "stds":   std_opt_list,
        },
    }

    out_path = path / "deploy.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
