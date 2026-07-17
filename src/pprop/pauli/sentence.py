"""
This module defines :class:`PauliDict`, a mapping from :class:`~pprop.pauli.op.PauliOp`
to a list of trigonometric coefficient terms.

Coefficient representation
--------------------------
Each coefficient is stored as a :data:`CoeffTerms`, 
a list of :data:`CoeffTerm` tuples of the form
``(coeff, sin_idx, cos_idx)``, encoding the product:

.. math::

    c \\prod_{i \\in \\text{sin\\_idx}} \\sin(\\theta_i)
      \\prod_{j \\in \\text{cos\\_idx}} \\cos(\\theta_j)

"""
from __future__ import annotations

from typing import ItemsView, KeysView, ValuesView

from numpy import integer, intp
from pennylane.ops.op_math import sum as qml_sum

from .op import PauliOp

# A single trigonometric product term:  coeff * ∏ sin(θᵢ) * ∏ cos(θⱼ)
CoeffTerm  = tuple[float, list[int], list[int]]   # (scalar, sin_indices, cos_indices)

# The full coefficient of one PauliOp: a sum of CoeffTerms.
CoeffTerms = list[CoeffTerm]

class PauliDict:
    """
    A mapping from :class:`~pprop.pauli.op.PauliOp` to :data:`CoeffTerms`.

    Each :class:`~pprop.pauli.op.PauliOp` key maps to a *list* of
    :data:`CoeffTerm` tuples, where each tuple encodes one trigonometric
    product term. The full coefficient at parameters
    :math:`\\boldsymbol{\\theta}` is:

    .. math::

        \\sum_k c_k
            \\prod_{i \\in S_k} \\sin(\\theta_i)
            \\prod_{j \\in C_k} \\cos(\\theta_j)

    where :math:`(c_k, S_k, C_k)` ranges over the list stored for that key.

    Parameters
    ----------
    data : dict, optional
        Initial mapping of ``PauliOp -> CoeffTerms``. If ``None`` (default),
        an empty dict is used.

    Examples
    --------
    >>> d = PauliDict()
    >>> key = PauliOp(0b01, 0b00)    # X on qubit 0
    >>> d.add_term(key, (0.5, [0], [1]))    # 0.5 * sin(θ₀) * cos(θ₁)
    >>> d.add_term(key, (0.5, [], [0, 1]))  # 0.5 * cos(θ₀) * cos(θ₁)
    """

    __slots__ = ("_dict", "_qubit_refcount", "_active_mask")

    def __init__(self, data: dict | None = None) -> None:
        self._dict: dict[PauliOp, CoeffTerms] = dict(data) if data is not None else {}
        # Per-qubit reference counts and their OR-aggregate (`_active_mask`):
        # how many keys currently touch each qubit, so `active_mask` - "which
        # qubits does *something* in this dict act non-trivially on" - stays
        # an O(1) read instead of an O(len(self)) scan over every key. Used by
        # heisenberg() to skip gates whose wires can't affect anything
        # currently tracked. Incremented/decremented only on key
        # insertion/removal (not on updates to an existing key's CoeffTerms),
        # so it costs O(popcount(key mask)) - not O(len(self)) - per mutation.
        self._qubit_refcount: dict[int, int] = {}
        self._active_mask = 0
        for key in self._dict:
            self._add_key_refs(key)

    def _add_key_refs(self, key: PauliOp) -> None:
        """Register a newly-inserted key's qubits in the refcount/active-mask."""
        mask = key.x | key.z
        m = mask
        while m:
            low = m & (-m)
            q = low.bit_length() - 1
            self._qubit_refcount[q] = self._qubit_refcount.get(q, 0) + 1
            m &= m - 1
        self._active_mask |= mask

    def _remove_key_refs(self, key: PauliOp) -> None:
        """Unregister a removed key's qubits from the refcount/active-mask."""
        mask = key.x | key.z
        m = mask
        while m:
            low = m & (-m)
            q = low.bit_length() - 1
            count = self._qubit_refcount.get(q, 0) - 1
            if count <= 0:
                self._qubit_refcount.pop(q, None)
                self._active_mask &= ~low
            else:
                self._qubit_refcount[q] = count
            m &= m - 1

    @property
    def active_mask(self) -> int:
        """
        Bitmask OR of ``(key.x | key.z)`` over every key currently in this dict.

        Bit ``q`` is set iff at least one tracked Pauli word acts
        non-trivially (X, Y, or Z) on qubit ``q``. A gate whose wires don't
        intersect this mask would act as the identity on every term
        currently in the dict - see ``heisenberg()`` in
        :mod:`~pprop.propagator.evolve`.
        """
        return self._active_mask

    def __setitem__(self, key: PauliOp, value: CoeffTerms) -> None:
        """
        Set ``key`` to ``value``, replacing any existing entry.

        Parameters
        ----------
        key : PauliOp
        value : CoeffTerms
        """
        if key not in self._dict:
            self._add_key_refs(key)
        self._dict[key] = value

    def __getitem__(self, key: PauliOp) -> CoeffTerms:
        """
        Return the :data:`CoeffTerms` associated with ``key``.

        Parameters
        ----------
        key : PauliOp

        Returns
        -------
        CoeffTerms

        Raises
        ------
        KeyError
            If ``key`` is not present.
        """
        return self._dict[key]

    def __contains__(self, key: PauliOp) -> bool:
        """Return ``True`` if ``key`` is present in the mapping."""
        return key in self._dict

    def __len__(self) -> int:
        """Return the number of distinct :class:`~pprop.pauli.op.PauliOp` keys."""
        return len(self._dict)

    def __repr__(self) -> str:
        """
        Return a human-readable representation of the mapping.

        For mappings with fewer than 100 keys, each Pauli word and its
        coefficient terms are printed explicitly. For larger mappings a
        compact summary is returned instead to avoid flooding the terminal.

        Returns
        -------
        str
        """
        if len(self._dict) < 100:
            parts = []
            for k, terms in self._dict.items():
                term_strs = []
                for coeff, sin_idxs, cos_idxs in terms:
                    s = f"{coeff:.2f}"
                    for sin_idx in sin_idxs:
                        s += f"*sin(θ_{int(sin_idx)})" if isinstance(sin_idx, (integer, intp)) else f"*sin({sin_idx})"
                    for cos_idx in cos_idxs:
                        s += f"*cos(θ_{int(cos_idx)})" if isinstance(cos_idx, (integer, intp)) else f"*cos({cos_idx})"
                    term_strs.append(s)
                parts.append(f"({' + '.join(term_strs)})*{k}")
            return " + ".join(parts)
        # Fall back to a compact summary for large dicts.
        return f"PauliDict({len(self._dict)} terms)"

    def items(self) -> ItemsView[PauliOp, CoeffTerms]:
        """Return a view of ``(PauliOp, CoeffTerms)`` pairs."""
        return self._dict.items()

    def keys(self) -> KeysView[PauliOp]:
        """Return a view of all :class:`~pprop.pauli.op.PauliOp` keys."""
        return self._dict.keys()

    def values(self) -> ValuesView[CoeffTerms]:
        """Return a view of all :data:`CoeffTerms` values."""
        return self._dict.values()

    def add_term(self, key: PauliOp, term: CoeffTerm) -> None:
        """
        Append a single :data:`CoeffTerm` to the list for ``key``.

        This is the primary accumulation method during Heisenberg propagation:
        each evolved term is appended without any simplification.

        Parameters
        ----------
        key : PauliOp
            The Pauli word to which the term belongs.
        term : CoeffTerm
            A ``(coeff, sin_indices, cos_indices)`` tuple to append.
        """
        if key in self._dict:
            self._dict[key].append(term)
        else:
            self._add_key_refs(key)
            self._dict[key] = [term]

    def add_terms(self, key: PauliOp, terms: CoeffTerms) -> None:
        """
        Extend the coefficient list for ``key`` with multiple :data:`CoeffTerm` tuples.

        Parameters
        ----------
        key : PauliOp
            The Pauli word to update.
        terms : CoeffTerms
            A list of ``(coeff, sin_indices, cos_indices)`` tuples to append.
        """
        if key in self._dict:
            self._dict[key].extend(terms)
        else:
            self._add_key_refs(key)
            self._dict[key] = list(terms)

    def add_terms_from_dict(self, other: PauliDict) -> None:
        """
        Merge all entries from ``other`` into ``self``.

        For keys present in both dicts the term lists are concatenated;
        for keys only in ``other`` a copy of their list is inserted.

        Parameters
        ----------
        other : PauliDict
            The source mapping to merge from.
        """
        for k, terms in other._dict.items():
            if k in self._dict:
                self._dict[k].extend(terms)
            else:
                self._add_key_refs(k)
                self._dict[k] = list(terms)

    def remove_keys_from_dict(self, other: PauliDict) -> None:
        """
        Remove all keys from ``self`` that also appear in ``other``.

        Used by :meth:`__isub__` to discard Pauli words that have been
        replaced by their evolved counterparts during propagation.

        Parameters
        ----------
        other : PauliDict
            Keys to remove.
        """
        other_keys = other._dict.keys()
        new_dict = {}
        for k, v in self._dict.items():
            if k in other_keys:
                self._remove_key_refs(k)
            else:
                new_dict[k] = v
        self._dict = new_dict

    def __iadd__(self, other: PauliDict) -> PauliDict:
        """
        Merge ``other`` into ``self`` in-place (``self += other``).

        Parameters
        ----------
        other : PauliDict

        Returns
        -------
        PauliDict
            ``self``, updated in-place.
        """
        if other._dict:
            self.add_terms_from_dict(other)
        return self

    def __isub__(self, other: PauliDict) -> PauliDict:
        """
        Remove all keys of ``other`` from ``self`` in-place (``self -= other``).

        Equivalent to calling :meth:`remove_keys_from_dict`. Used during
        propagation to drop original Pauli words after they have been evolved.

        Parameters
        ----------
        other : PauliDict

        Returns
        -------
        PauliDict
            ``self``, updated in-place.
        """
        self.remove_keys_from_dict(other)
        return self

    @classmethod
    def from_qml(cls, qml_op) -> PauliDict:
        """
        Construct a :class:`PauliDict` from a PennyLane operator.

        The operator is decomposed into a sum of Pauli words via
        :func:`pennylane.ops.op_math.sum`. Each Pauli word receives a
        constant (parameter-independent) coefficient, encoded as a
        :data:`CoeffTerm` with empty ``sin_idx`` and ``cos_idx`` lists.

        Parameters
        ----------
        qml_op : pennylane.operation.Operator
            A PennyLane observable, typically the output of ``qml.expval(...)``.

        Returns
        -------
        PauliDict
        """
        result = cls()
        for c, w in zip(*qml_sum(qml_op).terms()):
            # Constant coefficients have no sin/cos dependence, so both index
            # lists are empty, this is a valid CoeffTerm with frequency 0.
            result.add_term(PauliOp.from_qml(w), (float(c), [], []))
        return result

    def copy(self) -> PauliDict:
        """
        Return a shallow copy of this PauliDict.

        Each key's CoeffTerms list is copied, but the tuples inside
        (which are immutable) are not duplicated.

        Returns
        -------
        PauliDict
        """
        return PauliDict({k: list(v) for k, v in self._dict.items()})