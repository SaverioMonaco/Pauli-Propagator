"""Deploy a trained MNIST quantum circuit to IBM Quantum hardware.

Loads a trained parameter vector from a ``result.json`` file, converts the
PennyLane circuit to OpenQASM 2, transpiles it for the target backend, and
submits a sampling job via the Qiskit Runtime Sampler.  The resulting
bitstrings and counts are saved alongside the input file as ``samples.json``.

Usage
-----
    python deploy.py --path /path/to/run/dir --num_shots 64
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pennylane as qml
from qiskit import qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import SamplerV2 as Sampler

sys.path.append('./')
from ansatz import ansatz as penny_circuit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy a trained MNIST circuit to IBM Quantum hardware."
    )
    parser.add_argument(
        "--path", type=str, required=True,
        help="Path to the result directory containing result.json.",
    )
    parser.add_argument(
        "--num_shots", type=int, default=64,
        help="Number of measurement shots (default: 64).",
    )
    parser.add_argument(
        "--backend", type=str, default="ibm_torino",
        help="IBM backend name (default: ibm_torino).",
    )
    parser.add_argument(
        "--channel", type=str, default="ibm_cloud",
        help="IBM Runtime channel (default: ibm_cloud).",
    )
    args = parser.parse_args()

    path      = Path(args.path)
    num_shots = args.num_shots

    # ------------------------------------------------------------------ #
    # Load trained parameters.
    # ------------------------------------------------------------------ #
    with open(path / "result.json") as f:
        data = json.load(f)
    params     = data["params"]
    num_qubits = data["num_qubits"]

    print("deploy.py")
    print(f"  path:       {path}")
    print(f"  num_shots:  {num_shots}")
    print(f"  num_params: {len(params)}")

    # ------------------------------------------------------------------ #
    # Build the PennyLane circuit and export to OpenQASM 2.
    # ------------------------------------------------------------------ #
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        penny_circuit(params, int(np.sqrt(num_qubits)))
        return qml.sample()

    openqasm_circuit = qml.to_openqasm(circuit)
    qiskit_circuit   = qasm2.loads(openqasm_circuit())
    qiskit_circuit.measure_all()

    # ------------------------------------------------------------------ #
    # Select the least-busy IBM backend with enough qubits.
    # ------------------------------------------------------------------ #
    backend = QiskitRuntimeService(channel=args.channel).backend(args.backend)
    print(f"  backend:    {backend.name}")

    # ------------------------------------------------------------------ #
    # Transpile for the target backend and run.
    # ------------------------------------------------------------------ #
    pm          = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_circuit = pm.run(qiskit_circuit)

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session)
        job     = sampler.run([isa_circuit], shots=num_shots)
        result  = job.result()

    # ------------------------------------------------------------------ #
    # Extract and save bitstrings.
    # ------------------------------------------------------------------ #
    pub_result = result[0]
    counts     = pub_result.data.meas.get_counts()
    bitstrings = pub_result.data.meas.get_bitstrings()

    output = {
        "job_id":     job.job_id(),
        "backend":    backend.name,
        "shots":      num_shots,
        "counts":     counts,
        "bitstrings": bitstrings,
    }

    out_file = path / "samples.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Results saved to {out_file}")


if __name__ == "__main__":
    main()
