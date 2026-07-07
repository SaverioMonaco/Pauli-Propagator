"""
This module defines :class:`Gate`, the abstract base class for all quantum gates
in the Pauli propagation framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from numpy import integer, intp
from pennylane.operation import Operation

from ..pauli.op import PauliOp
from ..pauli.sentence import CoeffTerms, PauliDict


class Gate(ABC):
    """
    Abstract base class for all quantum gates.

    Each concrete gate subclass stores a PennyLane operator instance for
    circuit drawing and a Heisenberg evolution rule used during Pauli
    propagation. The constructor validates that the number of wires and
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

    @abstractmethod
    def evolve(self, word: Tuple[PauliOp, CoeffTerms]) -> PauliDict:
        """
        Heisenberg-evolve a Pauli word through this gate.

        Computes :math:`G^\\dagger\\, P\\, G` for the Pauli word :math:`P`
        encoded in ``word``, where :math:`G` is this gate. The result is
        returned as a :class:`~pprop.pauli.sentence.PauliDict` mapping each
        output Pauli word to its updated :data:`~pprop.pauli.sentence.CoeffTerms`.

        Parameters
        ----------
        word : tuple[PauliOp, CoeffTerms]
            ``(pauliop, coeff_terms)`` pair representing the Pauli word to evolve
            and its current symbolic coefficient.

        Returns
        -------
        PauliDict
            The evolved Pauli word(s) with updated coefficients.
        """

    def __repr__(self) -> str:
        """
        Return a concise string representation of the gate.

        Returns
        -------
        str
            ``"GateName([wires])"`` for non-parametrised gates, or
            ``"GateName(θ_i, [wires])"`` for parametrised gates.
        """
        if self.parameter is None:
            return f"{self.qml_gate.name}({self.wires})"
        elif isinstance(self.parameter, (integer, intp)):
            return f"{self.qml_gate.name}({self.parameter}, {self.wires})"
        else:
            return f"{self.qml_gate.name}(θ_{self.parameter}, {self.wires})"