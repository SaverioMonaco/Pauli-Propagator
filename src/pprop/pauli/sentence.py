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

Only used to build each observable's *initial* term (via :meth:`PauliDict.from_qml`)
before handing it to the Rust extension ``pprop_rs``. Heisenberg evolution itself
happens entirely in Rust, so this class carries no mutation/merge machinery beyond
what constructing that initial term needs.
"""
from __future__ import annotations

from typing import ItemsView

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

    __slots__ = ("_dict",)

    def __init__(self, data: dict | None = None) -> None:
        self._dict: dict[PauliOp, CoeffTerms] = dict(data) if data is not None else {}

    def items(self) -> ItemsView[PauliOp, CoeffTerms]:
        """Return a view of ``(PauliOp, CoeffTerms)`` pairs."""
        return self._dict.items()

    def add_term(self, key: PauliOp, term: CoeffTerm) -> None:
        """
        Append a single :data:`CoeffTerm` to the list for ``key``.

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
            self._dict[key] = [term]

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
