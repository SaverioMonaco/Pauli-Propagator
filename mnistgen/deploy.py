"""
Sample a trained circuit on IBM Quantum hardware.

Reads ``circuit.qasm`` from a training run (written by train.py), transpiles
it for the chosen backend and draws ``--shots`` bitstrings with SamplerV2.
Needs ``qiskit`` and ``qiskit-ibm-runtime`` plus saved IBM credentials.

Output
------
    <run>/samples.json   {"backend", "job_id", "shots", "bitstrings", "counts"}

Bitstrings are Qiskit-ordered (qubit 0 is the rightmost character); use
``pprop.mnist.bitstrings_to_array`` to turn them into pixel arrays.

Usage
-----
    python deploy.py results/side8_d0123456789_obs1000_lr5e-05_k12/000 --backend ibm_berlin --shots 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from qiskit import qasm2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run", type=Path, help="run directory containing circuit.qasm")
    p.add_argument("--backend", default="ibm_torino")
    p.add_argument("--channel", default="ibm_cloud")
    p.add_argument("--shots", type=int, default=1024)
    p.add_argument("--optimization_level", type=int, default=1)
    args = p.parse_args()

    circuit = qasm2.loads((args.run / "circuit.qasm").read_text())
    backend = QiskitRuntimeService(channel=args.channel).backend(args.backend)
    pm = generate_preset_pass_manager(backend=backend, optimization_level=args.optimization_level)
    isa_circuit = pm.run(circuit)
    print(f"{circuit.num_qubits} qubits on {backend.name}, "
          f"transpiled depth {isa_circuit.depth()}, {args.shots} shots")

    job = SamplerV2(mode=backend).run([isa_circuit], shots=args.shots)
    print(f"job {job.job_id()} submitted, waiting ...")
    data = job.result()[0].data.c  # classical register 'c' from circuit.qasm

    out = args.run / "samples.json"
    out.write_text(json.dumps({
        "backend": backend.name,
        "job_id": job.job_id(),
        "shots": args.shots,
        "bitstrings": data.get_bitstrings(),
        "counts": data.get_counts(),
    }, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
