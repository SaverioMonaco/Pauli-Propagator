"""
Export a trained circuit to OpenQASM 3.0.

The ansatz is the same callable you hand to :class:`~pprop.Propagator`; it is
recorded once with the trained parameter values and every gate is written out
as a ``stdgates.inc`` instruction. No Qiskit is needed to produce the string;
:func:`to_qiskit` is a thin convenience for hardware runs.

Example
-------
>>> from pprop.qasm import to_qasm
>>> def ansatz(params):
...     qml.RX(params[0], wires=0)
...     qml.CNOT(wires=[0, 1])
>>> print(to_qasm(ansatz, [0.3]))
OPENQASM 3.0;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
rx(0.3) q[0];
cx q[0],q[1];
c = measure q;
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from pennylane.tape import QuantumScript, make_qscript

#: PennyLane operation name -> OpenQASM 3 (stdgates.inc) gate name.
#: Covers every gate :class:`~pprop.Propagator` understands.
GATE_NAMES = {
    "Hadamard": "h",
    "H": "h",
    "S": "s",
    "SX": "sx",
    "T": "t",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CNOT": "cx",
    "CY": "cy",
    "CZ": "cz",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
    "SWAP": "swap",
}


def _record(ansatz, params) -> QuantumScript:
    """Run the ansatz once with the given (trained) parameter values."""
    # A Propagator carries its ansatz; accept either.
    ansatz = getattr(ansatz, "ansatz", ansatz)
    # make_qscript (unlike a QuantumTape context) does not leak into an
    # enclosing queue, so this is safe to call inside a QNode.
    return make_qscript(ansatz)(np.asarray(params, dtype=float))


def _qubit_index(tape: QuantumScript) -> tuple[dict, int]:
    """Map PennyLane wire labels to QASM qubit indices.

    Integer wires keep their number (so ``wires=5`` is ``q[5]``, which keeps
    pixel <-> qubit bookkeeping intact); any other labels are numbered in the
    order they first appear on the tape.
    """
    wires = list(tape.wires)
    if all(isinstance(w, (int, np.integer)) and w >= 0 for w in wires):
        return {w: int(w) for w in wires}, max(wires) + 1
    return {w: i for i, w in enumerate(wires)}, len(wires)


def to_qasm(
    ansatz: Callable,
    params: Sequence[float],
    *,
    measure: bool = True,
) -> str:
    """Return the circuit ``ansatz(params)`` as an OpenQASM 3.0 string.

    Parameters
    ----------
    ansatz : callable or Propagator
        ``ansatz(params)`` queues PennyLane gates (measurements it returns
        are ignored). A :class:`~pprop.Propagator` is accepted too and its
        ``ansatz`` is used.
    params : array-like
        Trained parameter values, indexed exactly as in the ansatz.
    measure : bool
        Append ``c = measure q;`` (computational-basis readout of every qubit).
    """
    tape = _record(ansatz, params)
    index, num_qubits = _qubit_index(tape)

    lines = [
        "OPENQASM 3.0;",
        'include "stdgates.inc";',
        f"qubit[{num_qubits}] q;",
    ]
    if measure:
        lines.append(f"bit[{num_qubits}] c;")

    for op in tape.operations:
        qubits = ",".join(f"q[{index[w]}]" for w in op.wires)
        if op.name == "Barrier":
            lines.append(f"barrier {qubits};")
            continue
        if op.name not in GATE_NAMES:
            raise ValueError(f"No OpenQASM 3 mapping for gate {op.name!r}")
        angles = ",".join(repr(float(p)) for p in op.parameters)
        gate = GATE_NAMES[op.name] + (f"({angles})" if angles else "")
        lines.append(f"{gate} {qubits};")

    if measure:
        lines.append("c = measure q;")
    return "\n".join(lines) + "\n"


def save_qasm(ansatz: Callable, params: Sequence[float], path: str | Path, **kwargs) -> Path:
    """Write :func:`to_qasm` output to ``path`` and return the path."""
    path = Path(path)
    path.write_text(to_qasm(ansatz, params, **kwargs))
    return path


def to_qiskit(ansatz: Callable, params: Sequence[float], **kwargs):
    """Return the circuit as a ``qiskit.QuantumCircuit``.

    Requires the optional ``qiskit`` extra: ``pip install "pauli-propagator[qiskit]"``.
    """
    try:
        from qiskit import qasm3
        import qiskit_qasm3_import  # noqa: F401  (backend of qasm3.loads)
    except ImportError as err:
        raise ImportError(
            'to_qiskit needs qiskit; install it with pip install "pauli-propagator[qiskit]"'
        ) from err

    return qasm3.loads(to_qasm(ansatz, params, **kwargs))
