"""
This module defines :class:`Gate`, the base class for all quantum gates in
the Pauli propagation framework.

Each concrete gate subclass (see ``pprop/gates/*.py``) still declares its
Heisenberg evolution ``rule`` table as the human-readable reference/spec.
``tests/test_backends.py`` checks the Rust extension ``pprop_rs`` against
PennyLane directly on full random circuits; ``tests/test_rule_tables.py``
checks each gate's ``rule`` dict here against its Rust counterpart
(``native/pprop_rs/src/lib.rs``) entry-for-entry, via the
``evolve_single_gate_debug`` debug hook, so the two can't silently drift
apart. Actually *executing* that evolution happens only in Rust; ``Gate``
itself carries no ``evolve()`` method.
"""
from __future__ import annotations

from typing import List, Optional

from pennylane.operation import Operation


class Gate:
    """
    Base class for all quantum gates.

    Each concrete gate subclass stores a PennyLane operator instance for
    circuit drawing and metadata (wires, parameter index) used to build the
    gate list :meth:`~pprop.propagator.Propagator.propagate` hands to the
    Rust extension. The constructor validates that the number of wires and
    the presence or absence of a parameter are consistent with the
    PennyLane gate's expectations.

    Parameters
    ----------
    qml_gate : pennylane.operation.Operation
        PennyLane gate *class* (not instance) corresponding to this gate.
        The constructor instantiates it with a placeholder parameter value
        of ``1`` (for parametrised gates) or without parameters (for
        non-parametrised gates).
    wires : list[int]
        Qubit indices on which this gate acts.
    parameter : float, int, optional
        If it is np.intp or np.integer it represents the tndex of
        :math:`\\theta` in the global parameter vector.
        If it is float, it is actually an assigned value to the gate.
        If it is None, the gate is non-parametrised.
    Attributes
    ----------
    qml_gate : pennylane.operation.Operation
        Instantiated PennyLane operator, used for circuit drawing.
    wires : list[int]
        Qubit indices on which this gate acts.
    parameter : int or None
        Value of the parametrized gate if float, 
        index into the global parameter vector if int,
        or ``None`` for non-parametrised gates.

    Raises
    ------
    ValueError
        If the number of wires does not match the gate's requirement.
    ValueError
        If the gate expects more than one parameter (unsupported).
    ValueError
        If a parametrised gate is constructed without a ``parameter_index``.
    ValueError
        If a non-parametrised gate is constructed with a ``parameter_index``.
    """

    def __init__(
        self,
        qml_gate: Operation,
        wires: List[int],
        parameter: Optional[int] = None,
    ) -> None:
        # Instantiate the PennyLane gate with a placeholder value so we can
        # query its metadata (num_wires, num_params, name).
        self.qml_gate = (
            qml_gate(1, wires=wires) if parameter is not None
            else qml_gate(wires=wires)
        )
        self.wires = wires
        self.parameter = parameter

        # ------------------------------------------------------------------ #
        # Validation                                                           #
        # ------------------------------------------------------------------ #

        # 1. Wire count must match the gate's requirements.
        num_wires_expected = self.qml_gate.num_wires
        if len(wires) != num_wires_expected:
            raise ValueError(
                f"{self.qml_gate.name} requires {num_wires_expected} wire(s), "
                f"but {len(wires)} were given."
            )

        # 2. Only 0- or 1-parameter gates are supported.
        if self.qml_gate.num_params > 1:
            raise ValueError(
                f"{self.qml_gate.name} expects more than 1 parameter; "
                f"only 0- or 1-parameter gates are supported."
            )

        gate_has_param = self.qml_gate.num_params > 0

        # 3. Parametrised gate must receive a parameter_index.
        if gate_has_param and parameter is None:
            raise ValueError(
                f"{self.qml_gate.name} requires a parameter, "
                f"but parameter is None."
            )

        # 4. Non-parametrised gate must not receive a parameter_index.
        if not gate_has_param and parameter is not None:
            raise ValueError(
                f"{self.qml_gate.name} does not accept parameters, "
                f"but parameter_index={parameter} was given."
            )