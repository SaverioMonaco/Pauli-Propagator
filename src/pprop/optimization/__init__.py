"""
This module provides an Adam optimiser for minimising a loss of the form
:math:`L(f(\\boldsymbol{\\theta}))`, where :math:`f` is a
:class:`~pprop.propagator.Propagator` and :math:`L` is a user-supplied scalar
loss function.

Gradients are computed via the chain rule:

.. math::

    \\frac{\\partial L}{\\partial \\boldsymbol{\\theta}}
    = \\frac{\\partial L}{\\partial \\mathbf{f}}
      \\cdot \\frac{\\partial \\mathbf{f}}{\\partial \\boldsymbol{\\theta}}

where :math:`\\partial \\mathbf{f}/\\partial \\boldsymbol{\\theta}` is
obtained analytically from
:meth:`~pprop.propagator.Propagator.eval_and_grad`, and
:math:`\\partial L / \\partial \\mathbf{f}` is either supplied directly by
the caller via ``grad_L``, or estimated by central finite differences via
:func:`_numerical_grad`.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import optax


def adam(
    L: Callable[[np.ndarray], float],
    propagator,
    params_init: np.ndarray,
    lr: float = 1e-3,
    num_steps: int = 1000,
    print_every: int = 100,
    grad_L: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> dict:
    """
    Minimize :math:`L(f(\\boldsymbol{\\theta}))` using the Adam optimiser.

    At each step the gradient is assembled via the chain rule:

    .. math::

        \\nabla_{\\boldsymbol{\\theta}} L
        = \\underbrace{\\nabla_{\\mathbf{f}} L}_{\\text{grad\\_L or finite diff.}}
          \\cdot
          \\underbrace{\\frac{\\partial \\mathbf{f}}{\\partial \\boldsymbol{\\theta}}}_{\\text{analytic}}

    The gradient :math:`\\nabla_{\\mathbf{f}} L` is computed in one of two ways:

    - If ``grad_L`` is provided, it is called directly. This is exact and
      efficient; a natural choice is ``jax.grad(L)`` when ``L`` is written
      with JAX-compatible operations.
    - If ``grad_L`` is ``None`` (default), the gradient is estimated by
      central finite differences via :func:`_numerical_grad`. This requires
      no assumptions on ``L`` beyond it being callable.

    Parameters
    ----------
    L : Callable[[ndarray], float]
        Scalar loss function. Receives ``f_vals`` of shape ``(num_obs,)``
        and returns a float.
    propagator : Propagator
        A propagated :class:`~pprop.propagator.Propagator` instance exposing
        an ``eval_and_grad(params)`` method.
    params_init : ndarray of shape (num_params,)
        Initial parameter vector. A copy is taken so the original is not modified.
    lr : float, optional
        Adam learning rate. Defaults to ``1e-3``.
    num_steps : int, optional
        Number of optimisation steps. Defaults to ``1000``.
    print_every : int, optional
        Print a progress line every this many steps. Set to ``0`` for silent
        operation. Defaults to ``100``.
    grad_L : Callable[[ndarray], ndarray], optional
        Gradient of ``L`` with respect to its input ``f_vals``. Should return
        an array of shape ``(num_obs,)``. If ``None``, central finite differences
        are used instead. A typical choice is ``jax.grad(L)`` when ``L`` is
        JAX-compatible.

    Returns
    -------
    dict with keys:

    ``params`` : ndarray of shape (num_params,)
        Final parameter vector after optimisation.
    ``fun`` : float
        Loss value at the final parameters.
    ``history`` : list[float]
        Loss value recorded at every step.

    Examples
    --------
    NumPy loss: finite differences used automatically:

    >>> result = adam(lambda f: float(np.sum(f**2)), propagator, params_init)

    JAX loss: exact gradient via ``jax.grad``:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> L_jax = lambda f: jnp.sum(f**2)
    >>> result = adam(L_jax, propagator, params_init, grad_L=jax.grad(L_jax))
    """
    optimizer = optax.adam(lr)

    params    = params_init.copy().astype(float)
    opt_state = optimizer.init(params)
    loss_history: list[float] = []
    params_history: list[float] = []

    # Build the gradient callable once outside the loop.
    # If the user supplies grad_L we use it directly; otherwise we wrap L
    # in a central finite-difference estimator.
    _grad_L: Callable[[np.ndarray], np.ndarray] = (
        grad_L if grad_L is not None else _numerical_grad(L)
    )

    for step in range(1, num_steps + 1):
        # Evaluate f(θ) and its Jacobian ∂f/∂θ analytically.
        f_vals, f_grads = propagator.eval_and_grad(params)  # (num_obs,), (num_obs, num_params)

        # Evaluate the scalar loss and ∂L/∂f.
        loss = float(L(f_vals))
        dLdf = _grad_L(f_vals)                              # (num_obs,)

        # Chain rule: ∂L/∂θ = (∂L/∂f) @ (∂f/∂θ)
        grad = dLdf @ f_grads                               # (num_params,)

        loss_history.append(loss)
        params_history.append(params.copy().tolist())

        # Apply one Adam step and update parameters.
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if print_every and step % print_every == 0:
            print(f"  step {step:5d}/{num_steps}  loss = {loss:.8f}")

    return {
        "params":  params.tolist(),
        "fun":     float(L(propagator(params))),
        "loss_history": loss_history,
        "params_history": params_history,
    }


def _numerical_grad(
    L: Callable[[np.ndarray], float],
    eps: float = 1e-5,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Return a central finite-difference gradient function for ``L``.

    For each component :math:`f_i`, the partial derivative is approximated as:

    .. math::

        \\frac{\\partial L}{\\partial f_i}
        \\approx \\frac{L(\\mathbf{f} + \\epsilon\\,\\mathbf{e}_i)
                      - L(\\mathbf{f} - \\epsilon\\,\\mathbf{e}_i)}{2\\epsilon}

    Parameters
    ----------
    L : Callable[[ndarray], float]
        Scalar loss function.
    eps : float, optional
        Finite-difference step size. Defaults to ``1e-5``.

    Returns
    -------
    Callable[[ndarray], ndarray]
        A function that accepts ``f_vals`` of shape ``(num_obs,)`` and returns
        the estimated gradient of the same shape.
    """
    def _grad(f_vals: np.ndarray) -> np.ndarray:
        g = np.zeros_like(f_vals)
        for i in range(len(f_vals)):
            fp = f_vals.copy()
            fp[i] += eps
            fm = f_vals.copy()
            fm[i] -= eps
            g[i] = (L(fp) - L(fm)) / (2 * eps)
        return g

    return _grad