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

Two entry points:

- :func:`adam` - works with any backend. Calls
  :meth:`~pprop.propagator.Propagator.eval_and_grad` once per step, which for
  ``backend in ("standard", "sparse")`` is already plain NumPy end to end -
  no host<->device transfer to eliminate.
- :func:`adam_gpu` - for ``backend="vmap"`` propagators specifically. Fuses
  the whole step (propagator forward+gradient, loss, Adam update) into one
  ``jax.lax.scan``-compiled program, so parameters never leave the device
  mid-run. ``adam``'s per-step ``jnp.asarray`` in / ``np.asarray`` out
  round-trip is cheap on its own, but once GPU compute itself drops to
  sub-millisecond (typical at the observable-batch sizes
  ``scripts/generation/train.py`` actually uses), that round-trip becomes
  most of the per-step cost - measured ~8.5x faster fused (including the
  one-time ``jax.lax.scan`` compile - see below), at ``train.py``'s default
  scale (``--num_obs 20``, ``--num_steps 5000``; see
  ``notebooks/test/gpu_backend.ipynb``), on a *cold* run: a fresh
  ``Propagator`` with no prior warm-up call, matching how
  ``scripts/generation/train.py`` actually calls this once per round (each
  round re-propagates a new observable batch, so there's no warm compiled
  function to carry over between rounds either way).
"""
from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.numpy as jnp
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


def adam_gpu(
    L: Callable[[jnp.ndarray], jnp.ndarray],
    propagator,
    params_init: np.ndarray,
    lr: float = 1e-3,
    num_steps: int = 1000,
    print_every: int = 100,
    grad_L: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None,
) -> dict:
    """
    JAX-native counterpart to :func:`adam`, for ``backend="vmap"`` propagators.

    Same math as :func:`adam` - the chain rule
    :math:`\\nabla_\\theta L = (\\nabla_f L) \\cdot (\\partial f/\\partial\\theta)`,
    minimised with Adam - but the entire loop (propagator forward pass +
    gradient, loss, ``grad_L``, and the Adam update) is fused into a single
    ``jax.jit`` + ``jax.lax.scan`` program. ``params``/the optimiser state
    never leave the device mid-run - only synced back to host once, after
    ``num_steps`` steps, instead of once per step as :func:`adam` does via
    :meth:`~pprop.propagator.Propagator.eval_and_grad`'s NumPy round-trip.
    That round-trip is what makes :func:`adam` disproportionately slow here:
    once GPU compute itself is sub-millisecond (typical for this project's
    per-round observable-batch sizes), it's most of the per-step cost -
    measured ~8.5x faster fused (a *cold* run - fresh ``Propagator``, one
    ``jax.lax.scan`` compile included, no prior warm-up call - since that's
    how ``scripts/generation/train.py`` actually calls this, once per round,
    each round re-propagating a new observable batch) at ``train.py``'s
    default scale (``--num_obs 20``, ``--num_steps 5000``). The advantage
    grows with steps-per-round, since the one-time compile is a fixed cost
    amortised over more steps - and shrinks towards it as steps-per-round
    drops (see ``notebooks/test/gpu_backend.ipynb`` for the numbers behind
    this).

    **Both `L` and `grad_L` must be written with `jax.numpy`, not `numpy`** -
    they get traced into the same compiled program as the propagator's own
    JAX computation, and a plain NumPy call on a traced array raises inside
    `jax.jit`. If you already have a NumPy version for :func:`adam`'s CPU
    path, porting it to ``jnp`` is usually a search-and-replace - the
    operations this kind of loss needs (elementwise arithmetic, ``dot``,
    ``sum``) exist under the same names in both.

    Parameters
    ----------
    L : Callable[[jnp.ndarray], jnp.ndarray]
        Scalar loss function, written with ``jax.numpy`` operations. Receives
        ``f_vals`` of shape ``(num_obs,)``.
    propagator : Propagator
        Must have been propagated with ``backend="vmap"`` - raises
        :exc:`ValueError` otherwise. ``"standard"``/``"sparse"`` are plain
        NumPy already, so there's no host<->device round-trip to eliminate,
        and no raw JAX function to fuse into - use :func:`adam` for those.
    params_init : ndarray of shape (num_params,)
        Initial parameter vector. Not modified in place.
    lr : float, optional
        Adam learning rate. Defaults to ``1e-3``.
    num_steps : int, optional
        Number of optimisation steps, all run as one compiled
        ``jax.lax.scan`` - unlike :func:`adam`, there is no partial/early
        readout mid-run. Defaults to ``1000``.
    print_every : int, optional
        Print a progress line every this many steps, indexed into the full
        loss history *after* the scan completes (there's no way to print
        mid-scan - it's one compiled call, not a Python loop). Set to ``0``
        for silent operation. Defaults to ``100``.
    grad_L : Callable[[jnp.ndarray], jnp.ndarray], optional
        Gradient of ``L`` with respect to its input ``f_vals``, written with
        ``jax.numpy``. If ``None`` (default), ``jax.grad(L)`` is used - exact
        autodiff, not the finite-difference fallback :func:`adam` uses when
        ``grad_L`` is omitted (which wouldn't survive being traced anyway).

    Returns
    -------
    dict with keys:

    ``params`` : ndarray of shape (num_params,)
        Final parameter vector after optimisation.
    ``fun`` : float
        Loss value at the final parameters.
    ``history`` : list[float]
        Loss value recorded at every step.

    Note there is no ``params_history`` key (unlike :func:`adam`) - keeping
    the full per-step parameter trajectory would force materialising every
    step's ``(num_params,)`` vector on the host, exactly the kind of
    per-step bookkeeping this function exists to avoid. Not used by
    ``scripts/generation/train.py`` in any case.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> mu, w = jnp.asarray(data_moments), jnp.asarray(weights)
    >>> L = lambda f: jnp.dot(w, (f - mu) ** 2)
    >>> result = adam_gpu(L, propagator, params_init, lr=0.05, num_steps=5000)
    """
    if propagator.backend != "vmap":
        raise ValueError(
            f"adam_gpu requires a Propagator propagated with backend='vmap' "
            f"(got backend={propagator.backend!r}). 'standard'/'sparse' are "
            "plain NumPy already - there's no host<->device round-trip to "
            "eliminate, and no raw JAX function to fuse into. Use adam() "
            "instead."
        )

    eval_and_grad = propagator._raw_eval_and_grad  # jnp in, jnp out - see vmap_backend.py
    _grad_L = grad_L if grad_L is not None else jax.grad(L)

    optimizer = optax.adam(lr)
    params0 = jnp.asarray(params_init)
    opt_state0 = optimizer.init(params0)

    def _step(carry, _):
        params, opt_state = carry
        f_vals, f_grads = eval_and_grad(params)              # (num_obs,), (num_obs, num_params)
        loss = L(f_vals)
        dLdf = _grad_L(f_vals)                                # (num_obs,)
        grad = dLdf @ f_grads                                 # (num_params,)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return (params, opt_state), loss

    @jax.jit
    def _run(params0, opt_state0):
        (params, opt_state), loss_history = jax.lax.scan(
            _step, (params0, opt_state0), xs=None, length=num_steps
        )
        return params, loss_history

    params_final, loss_history = _run(params0, opt_state0)

    # The only two host syncs in the whole run: here, and the final-loss
    # readout below - not one per step.
    params_final = np.asarray(params_final)
    loss_history = np.asarray(loss_history)

    if print_every:
        for step in range(print_every - 1, num_steps, print_every):
            print(f"  step {step + 1:5d}/{num_steps}  loss = {loss_history[step]:.8f}")

    final_vals, _ = eval_and_grad(jnp.asarray(params_final))
    return {
        "params": params_final.tolist(),
        "fun": float(L(final_vals)),
        "loss_history": loss_history.tolist(),
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