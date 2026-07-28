"""
Affine reparametrisation of a :class:`~pprop.propagator.Propagator`'s
``num_params``-sized parameter vector down to a smaller, user-defined "free"
vector, e.g. one gate reading ``f0`` and another reading ``-2*f0``.

:class:`Free` is a symbolic placeholder for one free parameter; combining
several with ``+``, ``-``, ``*``/``/`` by plain numbers builds an affine
expression (a sparse set of coefficients plus a constant offset). Passing one
such expression per trainable index to :meth:`Propagator.bind` assembles the
Jacobian ``J`` and offset ``b`` of ``theta = J @ free + b`` automatically; the
returned :class:`BoundPropagator` evaluates at ``theta`` and turns pprop's own
gradient w.r.t. ``theta`` into the gradient w.r.t. ``free`` via the chain
rule, ``grad_free = grad_theta @ J``. No approximation: this is exact for any
affine dependency between gate angles, since pprop's own gradients are already
exact closed-form calculus.
"""
from __future__ import annotations

from numbers import Number
from typing import Dict, List, Sequence, Union

from numpy import asarray, ndarray, zeros


class Free:
    """
    A symbolic placeholder for one free (independent, user-facing) parameter.

    Combine with ``+``, ``-``, ``*``, ``/`` and plain numbers to build an
    affine expression of one or more free parameters, then pass a list of
    such expressions (one per :attr:`Propagator.num_params` index, in gate
    order) to :meth:`Propagator.bind`.

    Examples
    --------
    >>> f0, f1 = Free.vars(2)
    >>> theta = [f0, -2 * f0, f1, f0 + 3 * f1 - 0.5]
    """

    __slots__ = ("_coeffs", "_offset")

    def __init__(self, coeffs: Dict[int, float] | None = None, offset: float = 0.0) -> None:
        self._coeffs: Dict[int, float] = dict(coeffs) if coeffs else {}
        self._offset: float = float(offset)

    @classmethod
    def vars(cls, n: int) -> List["Free"]:
        """Return ``n`` independent free-parameter symbols, ``f0 .. f(n-1)``."""
        return [cls({i: 1.0}) for i in range(n)]

    @staticmethod
    def _as_free(other: Union["Free", Number]) -> "Free":
        if isinstance(other, Free):
            return other
        return Free(offset=other)

    def __add__(self, other: Union["Free", Number]) -> "Free":
        other = self._as_free(other)
        coeffs = dict(self._coeffs)
        for k, v in other._coeffs.items():
            coeffs[k] = coeffs.get(k, 0.0) + v
        return Free(coeffs, self._offset + other._offset)

    __radd__ = __add__

    def __neg__(self) -> "Free":
        return Free({k: -v for k, v in self._coeffs.items()}, -self._offset)

    def __sub__(self, other: Union["Free", Number]) -> "Free":
        return self + (-self._as_free(other))

    def __rsub__(self, other: Union["Free", Number]) -> "Free":
        return self._as_free(other) + (-self)

    def __mul__(self, scalar: Number) -> "Free":
        if isinstance(scalar, Free):
            raise TypeError("Free * Free is not affine; only Free * number is supported.")
        return Free({k: v * scalar for k, v in self._coeffs.items()}, self._offset * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: Number) -> "Free":
        return self * (1.0 / scalar)

    def __repr__(self) -> str:
        terms = [f"{v:+g}*f{k}" for k, v in sorted(self._coeffs.items())]
        if self._offset or not terms:
            terms.append(f"{self._offset:+g}")
        return "Free(" + " ".join(terms).lstrip("+") + ")"


def affine_from_exprs(exprs: Sequence[Union[Free, Number]], num_params: int):
    """
    Assemble ``(J, b, num_free)`` for ``theta = J @ free + b`` from one
    :class:`Free` expression (or plain number, for a fixed value) per
    trainable index.

    Raises
    ------
    ValueError
        If ``len(exprs) != num_params``.
    """
    if len(exprs) != num_params:
        raise ValueError(
            f"Expected {num_params} entries (one per Propagator.num_params), "
            f"got {len(exprs)}."
        )
    resolved = [e if isinstance(e, Free) else Free(offset=e) for e in exprs]
    num_free = max((k for e in resolved for k in e._coeffs), default=-1) + 1

    J = zeros((num_params, num_free))
    b = zeros(num_params)
    for i, e in enumerate(resolved):
        for k, v in e._coeffs.items():
            J[i, k] = v
        b[i] = e._offset
    return J, b, num_free


class BoundPropagator:
    """
    An affine reparametrisation of a :class:`~pprop.propagator.Propagator`,
    built by :meth:`Propagator.bind`. Evaluates and differentiates directly
    in terms of the smaller ``free`` vector.

    Attributes
    ----------
    prop : Propagator
        The wrapped propagator, evaluated at ``J @ free + b``.
    J : numpy.ndarray of shape (num_params, num_free)
    b : numpy.ndarray of shape (num_params,)
    num_free : int
        Number of independent free parameters, ``J.shape[1]``.
    """

    def __init__(self, prop, J: ndarray, b: ndarray) -> None:
        self.prop = prop
        self.J = J
        self.b = b
        self.num_free = J.shape[1]

    def __call__(self, free: ndarray) -> ndarray:
        """Evaluate all observables at ``free``. See ``Propagator.__call__``."""
        return self.prop(self.J @ asarray(free) + self.b)

    def eval_and_grad(self, free: ndarray):
        """
        Evaluate values and gradients w.r.t. ``free``.

        Returns
        -------
        vals : ndarray of shape (num_observables,)
        grads : ndarray of shape (num_observables, num_free)
            ``d(vals)/d(free) = d(vals)/d(theta) @ J``, by the chain rule.
        """
        vals, grads = self.prop.eval_and_grad(self.J @ asarray(free) + self.b)
        return vals, grads @ self.J
