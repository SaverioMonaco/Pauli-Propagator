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

from pennylane import Identity, X, Y, Z


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

    Examples
    --------
    >>> PauliOp(0b101, 0b110)
    Y0 Z1 X2

    >>> PauliOp(0b1001)
    X0 X3

    >>> PauliOp()
    I
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

    def __getitem__(self, qubit: int) -> str:
        """
        Return the single-qubit Pauli operator at ``qubit``.

        Parameters
        ----------
        qubit : int
            Zero-based qubit index.

        Returns
        -------
        str
            One of ``"X"``, ``"Y"``, ``"Z"``, or ``"I"``.
        """
        x_bit = (self.x >> qubit) & 1
        z_bit = (self.z >> qubit) & 1
        if x_bit and z_bit:
            return "Y"
        elif x_bit:
            return "X"
        elif z_bit:
            return "Z"
        else:
            return "I"

    def set(self, qubit: int, op: str) -> None:
        """
        Set the Pauli operator on a specific qubit in-place.

        Updates the ``x`` and ``z`` bitmasks at bit position ``qubit`` to
        reflect the requested operator according to the bitmask convention.

        Parameters
        ----------
        qubit : int
            Zero-based qubit index to update.
        op : str
            Target operator; one of ``"I"``, ``"X"``, ``"Y"``, or ``"Z"``.

        Raises
        ------
        ValueError
            If ``op`` is not one of the four valid Pauli operators.
        """
        if op == "X":
            self.x |=  (1 << qubit)   # set x bit
            self.z &= ~(1 << qubit)   # clear z bit
        elif op == "Y":
            self.x |= (1 << qubit)    # set both bits
            self.z |= (1 << qubit)
        elif op == "Z":
            self.x &= ~(1 << qubit)   # clear x bit
            self.z |=  (1 << qubit)   # set z bit
        elif op == "I":
            self.x &= ~(1 << qubit)   # clear both bits
            self.z &= ~(1 << qubit)
        else:
            raise ValueError(f"Invalid Pauli operator '{op}'; expected 'I', 'X', 'Y', or 'Z'")

    def qubits(self) -> set[int]:
        """
        Return the set of qubits where this word acts non-trivially (not as I).

        Returns
        -------
        set[int]
            Qubit indices where ``self[k] != "I"``.
        """
        active = set()
        # Only need to inspect bits up to the highest set bit across both masks.
        n = max(int(self.x).bit_length(), int(self.z).bit_length())
        for i in range(n):
            if ((self.x >> i) & 1) or ((self.z >> i) & 1):
                active.add(i)
        return active

    def weight(self) -> int:
        """
        Return the Pauli weight, i.e. the number of non-identity single-qubit factors.

        Computed as the popcount of ``x | z``: a qubit is non-identity if and
        only if at least one of its ``x`` or ``z`` bits is set.

        Returns
        -------
        int
            Number of qubits where the operator is X, Y, or Z.
        """
        return (self.x | self.z).bit_count()

    def zerobracket(self) -> bool:
        """
        Return ``True`` if this Pauli word has zero expectation in all but the Z/I basis.

        A Pauli word :math:`P` satisfies :math:`\\langle 0 | P | 0 \\rangle \\neq 0`
        if and only if every single-qubit factor is either :math:`Z` or :math:`I`.
        This is equivalent to checking that no X bit is set (``x == 0``).

        Returns
        -------
        bool
        """
        return self.x == 0

    def zerobracket_X(self) -> bool:
        """
        Return True if this Pauli word has nonzero expectation in |+><+|,
        i.e. it is a pure X or I string (no Z or Y).

        Returns
        -------
        bool
        """
        return self.z == 0

    def copy(self) -> PauliOp:
        """
        Return a shallow copy of this :class:`PauliOp`.

        Returns
        -------
        PauliOp
            A new instance with identical ``x`` and ``z`` bitmasks.
        """
        return PauliOp(self.x, self.z)

    def to_qml(self, indices: list[int]):
        """
        Convert this :class:`PauliOp` to a PennyLane operator on a subset of qubits.

        Only the qubits listed in ``indices`` are included; qubits acting as
        identity are skipped. If all qubits are identity an
        :class:`~pennylane.Identity` on wire 0 is returned as a fallback.

        Parameters
        ----------
        indices : list[int]
            Qubit indices to include in the operator.

        Returns
        -------
        pennylane.operation.Operator
            Tensor product of single-qubit Pauli operators over ``indices``.
        """
        ops = []
        for k in indices:
            p = self[k]
            if p == "X":
                ops.append(X(k))
            elif p == "Y":
                ops.append(Y(k))
            elif p == "Z":
                ops.append(Z(k))
            # Identity factors are omitted from the tensor product.

        if not ops:
            return Identity(0)

        # Build the tensor product left-to-right using PennyLane's @ operator.
        result = ops[0]
        for op in ops[1:]:
            result @= op
        return result

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

    def __repr__(self) -> str:
        """
        Return a human-readable string representation of the Pauli word.

        Returns
        -------
        str
            Space-separated Pauli labels like ``"X0 Y2 Z3"``, or ``"I"`` for
            the identity word.
        """
        result = []
        # Inspect all bit positions up to the highest set bit in either mask.
        n_qubits = max(int(self.x).bit_length(), int(self.z).bit_length())
        for k in range(n_qubits):
            op = self[k]
            if op != "I":
                result.append(f"{op}{k}")
        return " ".join(result) if result else "I"