"""
The generative model: a shell-brick circuit on a ``side x side`` qubit grid.

Qubit ``r * side + c`` holds pixel ``(r, c)``. Layout (every layer covers
all qubits):

    RX  ->  horizontal CZ brickwork  ->  RX  ->  vertical CZ brickwork  ->  RX RY

The model's distribution is the computational-basis readout of this state;
training only ever needs its PauliZ-string expectation values.
"""

from __future__ import annotations

from functools import reduce

import pennylane as qml


def num_params(side: int) -> int:
    """Three single-rotation layers + one RY layer = 4 angles per qubit."""
    return 4 * side * side


def _cz_brickwork(side: int, horizontal: bool) -> None:
    """Two staggered layers of CZ between grid neighbours (even bonds, then odd)."""
    for offset in (0, 1):
        for line in range(side):
            for pos in range(offset, side - 1, 2):
                if horizontal:
                    a, b = line * side + pos, line * side + pos + 1
                else:
                    a, b = pos * side + line, (pos + 1) * side + line
                qml.CZ(wires=[a, b])


def ansatz(params, side: int) -> None:
    """Queue the shell-brick circuit; ``params`` has length ``num_params(side)``."""
    n = side * side
    rx1, rx2 = params[:n], params[n:2 * n]
    # Last layer is interleaved (RX_q, RY_q, RX_q+1, ...) to stay compatible
    # with parameter files from scripts/generation.
    rx3, ry = params[2 * n::2], params[2 * n + 1::2]

    for q in range(n):
        qml.RX(rx1[q], wires=q)
    _cz_brickwork(side, horizontal=True)

    for q in range(n):
        qml.RX(rx2[q], wires=q)
    _cz_brickwork(side, horizontal=False)

    for q in range(n):
        qml.RX(rx3[q], wires=q)
        qml.RY(ry[q], wires=q)


def z_string(qubits) -> qml.operation.Operator:
    """``Z_{q1} @ Z_{q2} @ ...`` on the given qubits."""
    return reduce(lambda a, b: a @ b, (qml.PauliZ(q) for q in qubits))


def circuit(side: int, observables=()):
    """Ansatz returning ``<O>`` for each observable: the callable a Propagator wants."""
    def _circuit(params):
        ansatz(params, side)
        return [qml.expval(ob) for ob in observables]
    return _circuit
