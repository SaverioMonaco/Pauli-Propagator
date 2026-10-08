"""
This module provides optimisers for minimising a loss of the form
:math:`L(f(\\boldsymbol{\\theta}))`, where :math:`f` is a
:class:`~pprop.propagator.Propagator` and :math:`L` is a user-supplied scalar
loss function: :func:`adam` (stochastic-style, fixed step size) and
:func:`lbfgs` (quasi-Newton with line search).

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

Both optimisers call :meth:`~pprop.propagator.Propagator.eval_and_grad` to
obtain a value/Jacobian pair, which is plain NumPy end to end since
propagation itself runs in the Rust extension ``pprop_rs`` (see
``pprop.propagator``). There's no host<->device transfer or JAX/GPU path in
pprop: the old ``adam_gpu``/``backend="vmap"`` combination was removed
along with the other propagation/evaluator backends.

Choosing between them
---------------------

After :meth:`~pprop.propagator.Propagator.propagate` has run, truncation and
pruning are already baked into the compiled expressions, so
:math:`f(\\boldsymbol{\\theta})` is a *fixed*, deterministic, analytic
trigonometric function and ``eval_and_grad`` returns its exact gradient. That
is the regime quasi-Newton methods are built for, so :func:`lbfgs` typically
reaches a given loss in far fewer evaluations than :func:`adam`, whose
per-coordinate step adaptation is designed for noisy stochastic gradients
that never arise here.

The trade-offs run the other way in two cases:

- **Local minima.** These landscapes are non-convex, and :func:`adam` with a
  small ``lr`` stays near its initial point, whereas :func:`lbfgs` drives
  hard into the nearest stationary point, which may be a worse one. Since an
  :func:`lbfgs` run is much cheaper, the usual answer is several random
  restarts rather than one long :func:`adam` run.
- **Flat regions.** :func:`adam` normalises by the gradient's running
  second moment, so it keeps taking ``lr``-sized steps even where
  :math:`\\|\\nabla L\\|` is vanishing, while :func:`lbfgs` will stop once
  ``gtol`` is met. On barren-plateau-prone ansätze, pass ``gtol=0.0`` to
  :func:`lbfgs` so only ``ftol``/``num_steps`` can end the run.
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional, Tuple

import numpy as np
import optax
from scipy.optimize import minimize


def _value_and_grad(
    L: Callable[[np.ndarray], float],
    propagator,
    grad_L: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Callable[[np.ndarray], Tuple[float, np.ndarray]]:
    """
    Build a ``theta -> (loss, dL/dtheta)`` callable from a loss and a propagator.

    One call to :meth:`~pprop.propagator.Propagator.eval_and_grad` yields both
    :math:`\\mathbf{f}` and :math:`\\partial\\mathbf{f}/\\partial\\boldsymbol{\\theta}`,
    which the chain rule combines with :math:`\\partial L/\\partial\\mathbf{f}`
    into the parameter gradient.

    Parameters
    ----------
    L : Callable[[ndarray], float]
        Scalar loss function of ``f_vals``.
    propagator : Propagator
        A propagated :class:`~pprop.propagator.Propagator` (or
        :class:`~pprop.propagator.binding.BoundPropagator`) exposing
        ``eval_and_grad(params)``.
    grad_L : Callable[[ndarray], ndarray], optional
        Gradient of ``L`` with respect to ``f_vals``. Falls back to
        :func:`_numerical_grad` when ``None``.

    Returns
    -------
    Callable[[ndarray], tuple[float, ndarray]]
        Function returning the loss and its gradient w.r.t. the parameters.
    """
    _grad_L: Callable[[np.ndarray], np.ndarray] = (
        grad_L if grad_L is not None else _numerical_grad(L)
    )

    def _f(params: np.ndarray) -> Tuple[float, np.ndarray]:
        # Evaluate f(θ) and its Jacobian ∂f/∂θ analytically.
        f_vals, f_grads = propagator.eval_and_grad(params)  # (num_obs,), (num_obs, num_params)

        # Evaluate the scalar loss and ∂L/∂f.
        loss = float(L(f_vals))
        dLdf = _grad_L(f_vals)                              # (num_obs,)

        # Chain rule: ∂L/∂θ = (∂L/∂f) @ (∂f/∂θ)
        return loss, dLdf @ f_grads                         # (num_params,)

    return _f


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
    lr : float or optax.Schedule, optional
        Adam learning rate. Defaults to ``1e-3``. Passed straight to
        :func:`optax.adam`, so any optax schedule works in place of a
        constant, e.g. ``optax.cosine_decay_schedule(0.1, num_steps)``. A
        decaying schedule is usually worth it: a constant ``lr`` large enough
        to traverse the landscape early is too large to settle at the end.
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

    # Build the value/gradient callable once outside the loop. If the user
    # supplies grad_L we use it directly; otherwise L is wrapped in a central
    # finite-difference estimator.
    value_and_grad = _value_and_grad(L, propagator, grad_L)

    for step in range(1, num_steps + 1):
        loss, grad = value_and_grad(params)

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


def lbfgs(
    L: Callable[[np.ndarray], float],
    propagator,
    params_init: np.ndarray,
    num_steps: int = 500,
    print_every: int = 100,
    grad_L: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    gtol: float = 1e-8,
    ftol: float = 1e-12,
    max_evals: Optional[int] = None,
    maxcor: int = 10,
) -> dict:
    """
    Minimize :math:`L(f(\\boldsymbol{\\theta}))` using L-BFGS-B.

    A drop-in alternative to :func:`adam` with the same loss/propagator
    contract and the same return keys. Both the value and the gradient come
    from a single :meth:`~pprop.propagator.Propagator.eval_and_grad` call per
    evaluation, which is exactly the ``jac=True`` interface
    :func:`scipy.optimize.minimize` expects.

    Unlike :func:`adam`, there is no learning rate: the step length is chosen
    by a Wolfe line search, and curvature is approximated from the history of
    recent gradients. See the module docstring for when to prefer which.

    .. note::

       Supplying an exact ``grad_L`` matters more here than it does for
       :func:`adam`. The finite-difference fallback returns a gradient that is
       not exactly the gradient of the function being evaluated, and the Wolfe
       line search is sensitive to that inconsistency near convergence — it
       may stop early reporting ``ABNORMAL_TERMINATION_IN_LNSRCH``. A warning
       is emitted when ``grad_L`` is omitted.

    .. note::

       No periodicity is imposed on the parameters, so the returned angles may
       drift outside :math:`[0, 2\\pi)`. This is harmless, but note that
       wrapping them afterwards is only valid when each parameter feeds a
       single rotation gate directly; under
       :meth:`~pprop.propagator.Propagator.bind` a free parameter scaled by
       :math:`c` has period :math:`2\\pi/c` instead.

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
    num_steps : int, optional
        Maximum number of L-BFGS *iterations*. Each iteration performs a line
        search and so may spend several function evaluations. Defaults to
        ``500``.
    print_every : int, optional
        Print a progress line every this many iterations. Set to ``0`` for
        silent operation. Defaults to ``100``.
    grad_L : Callable[[ndarray], ndarray], optional
        Gradient of ``L`` with respect to its input ``f_vals``. Should return
        an array of shape ``(num_obs,)``. If ``None``, central finite
        differences are used instead, subject to the caveat above.
    gtol : float, optional
        Stop once the gradient's sup-norm falls below this value. Defaults to
        ``1e-8``, well below SciPy's own ``1e-5`` default so that shallow
        regions are not mistaken for convergence. Pass ``0.0`` to disable the
        test entirely on barren-plateau-prone problems.
    ftol : float, optional
        Stop once the relative decrease in the loss falls below this value.
        Defaults to ``1e-12``.
    max_evals : int, optional
        Objective-evaluation budget passed to SciPy's ``maxfun`` option.
        The limit is checked between iterations, so a line search may exceed
        it. Counts value/Jacobian calls made by the optimizer; finite-difference
        calls to ``L`` and the final loss recomputation are not included.
        Defaults to ``5 * num_steps``.
    maxcor : int, optional
        Number of correction pairs retained to approximate the inverse
        Hessian. Larger values buy a better curvature model at
        ``O(maxcor * num_params)`` memory. Defaults to ``10``.

    Returns
    -------
    dict with keys:

    ``params`` : list[float]
        Final parameter vector after optimisation.
    ``fun`` : float
        Loss value at the final parameters.
    ``loss_history`` : list[float]
        Loss recorded at every *function evaluation*, line-search trials
        included. Unlike :func:`adam`, this is therefore not monotonic and its
        length does not equal ``num_steps``; it records work done rather than
        iterations taken.
    ``params_history`` : list[list[float]]
        Parameter vector at every function evaluation, aligned with
        ``loss_history``.
    ``nit`` : int
        Number of L-BFGS iterations performed.
    ``nfev`` : int
        Number of function/gradient evaluations performed.
    ``success`` : bool
        SciPy's convergence flag.
    ``message`` : str
        SciPy's termination message.

    Examples
    --------
    Exact gradient, the recommended path (here ``L`` picks one observable, so
    :math:`\\partial L/\\partial\\mathbf{f}` is a constant unit vector):

    >>> result = lbfgs(
    ...     L=lambda f: f[0],
    ...     propagator=prop,
    ...     params_init=np.random.rand(prop.num_params),
    ...     grad_L=lambda f: np.array([1.0]),
    ... )

    Several restarts, keeping the best minimum found:

    >>> runs = [lbfgs(L, prop, np.random.rand(prop.num_params), print_every=0,
    ...                grad_L=grad_L) for _ in range(10)]
    >>> best = min(runs, key=lambda r: r["fun"])
    """
    if grad_L is None:
        warnings.warn(
            "lbfgs() called without grad_L, so dL/df falls back to finite "
            "differences. The resulting gradient is not exactly consistent with "
            "the evaluated loss, which can stall the Wolfe line search near "
            "convergence. Supply grad_L for an exact gradient.",
            RuntimeWarning,
            stacklevel=2,
        )

    params = params_init.copy().astype(float)
    value_and_grad = _value_and_grad(L, propagator, grad_L)

    loss_history: list[float] = []
    params_history: list[list[float]] = []

    # SciPy drives the search, so history is recorded here, at every
    # evaluation, rather than once per iteration as in adam's explicit loop.
    def _objective(theta: np.ndarray):
        loss, grad = value_and_grad(theta)
        loss_history.append(loss)
        params_history.append(theta.copy().tolist())
        return loss, grad

    iteration = 0

    def _callback(_xk: np.ndarray) -> None:
        nonlocal iteration
        iteration += 1
        if print_every and iteration % print_every == 0:
            print(f"  iter {iteration:5d}/{num_steps}  loss = {loss_history[-1]:.8f}")

    res = minimize(
        _objective,
        params,
        jac=True,
        method="L-BFGS-B",
        callback=_callback if print_every else None,
        options={
            "maxiter": num_steps,
            "maxfun": max_evals if max_evals is not None else 5 * num_steps,
            "gtol": gtol,
            "ftol": ftol,
            "maxcor": maxcor,
        },
    )

    final_params = np.asarray(res.x, dtype=float)

    return {
        "params": final_params.tolist(),
        "fun": float(L(propagator(final_params))),
        "loss_history": loss_history,
        "params_history": params_history,
        "nit": int(res.nit),
        "nfev": int(res.nfev),
        "success": bool(res.success),
        "message": str(res.message),
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