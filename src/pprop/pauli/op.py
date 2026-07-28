"""
This module defines :class:`PauliOp`, which represents a Pauli word as a pair
of bitmasks encoding X, Y, Z, and I operators across an arbitrary number of qubits.

Bitmask convention
------------------
Each qubit ``k`` is represented by bit ``k`` (i.e. ``1 << k``) in two integers:

=====  =====  =========
``x``  ``z``  Operator
=====  =====  =========
0      0      I
1      0      X
0      1      Z
1      1      Y
=====  =====  =========
"""
from __future__ import annotations

from collections.abc import Iterable

from pennylane import X, Y, Z


class PauliOp:
    """
    A Pauli word represented as two integer bitmasks.

    Using bitmasks instead of lists or dicts allows :class:`PauliOp` objects
    to be hashed cheaply and compared in O(1), which is important because
    Pauli propagation creates a very large number of them.
    :attr:`__slots__` is used to minimise per-instance memory overhead.

    The encoding maps each qubit ``k`` to bit ``k`` in two integers ``x`` and
    ``z`` according to the table below:

    =====  =====  =========
    ``x``  ``z``  Operator
    =====  =====  =========
    0      0      I
    1      0      X
    0      1      Z
    1      1      Y
    =====  =====  =========

    Parameters
    ----------
    x : int, optional
        Bitmask encoding the qubits that carry an X or Y factor. Defaults to 0 (all identity).
    z : int, optional
        Bitmask encoding the qubits that carry a Z or Y factor. Defaults to 0 (all identity).
    """

    __slots__ = ("x", "z")

    def __init__(self, x: int = 0, z: int = 0) -> None:
        self.x = x
        self.z = z

    def __hash__(self) -> int:
        """
        Hash the Pauli word by its ``(x, z)`` bitmask pair.

        Allows :class:`PauliOp` to be used as a dictionary key in
        :class:`~pprop.pauli.sentence.PauliDict`.

        Returns
        -------
        int
        """
        return hash((self.x, self.z))

    def __eq__(self, other: object) -> bool:
        """
        Test equality with another :class:`PauliOp`.

        Parameters
        ----------
        other : object
            The object to compare against.

        Returns
        -------
        bool
            ``True`` if ``other`` is a :class:`PauliOp` with identical ``(x, z)`` masks.
        """
        if not isinstance(other, PauliOp):
            return NotImplemented
        return self.x == other.x and self.z == other.z

    @classmethod
    def from_qml(cls, qml_op) -> PauliOp:
        """
        Construct a :class:`PauliOp` from a PennyLane operator.

        Accepts either a single-qubit PennyLane operator or an iterable of them
        (e.g. the result of iterating over a tensor product).

        Parameters
        ----------
        qml_op : pennylane.operation.Operator or Iterable
            A PennyLane X, Y, Z, or Identity operator, or an iterable thereof.

        Returns
        -------
        PauliOp
            Bitmask representation of the input operator.

        Notes
        -----
        Identity operators are skipped; their bits remain 0, which is the
        correct encoding for I.
        """
        x_mask = 0
        z_mask = 0

        # Accept both a single operator and an iterable of operators.
        ops = qml_op if isinstance(qml_op, Iterable) else [qml_op]
        for op in ops:
            wire     = op.wires[0]   # each op acts on exactly one qubit
            cls_type = type(op)
            if cls_type is X:
                x_mask |= 1 << wire
            elif cls_type is Y:
                # Y = iXZ, so both bits are set
                x_mask |= 1 << wire
                z_mask |= 1 << wire
            elif cls_type is Z:
                z_mask |= 1 << wire
            # Identity: both bits stay 0, nothing to do.

        return cls(x=x_mask, z=z_mask)
