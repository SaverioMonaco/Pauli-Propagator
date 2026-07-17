"""
Core module with the Propagator. 
Propagator takes as an input a quantum circuit as a function of a list of parameters List[float] and returns 
the expectation value of an observable.
>>> from pprop import Propagator
>>> import pennylane as qml
>>> def ansatz(params):
...     qml.RX(params[0], wires=0)
...     qml.RX(params[1], wires=1)
...     qml.CNOT(wires = [0, 1])
...     qml.RY(params[2], wires=0)
...     qml.RY(params[3], wires=1)
...     return [qml.expval(qml.PauliZ(0))]
>>> prop = Propagator(ansatz, k1 = None, k2 = None)
"""
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from multiprocessing import get_context
from typing import Callable, List, Optional, Tuple

from numpy import arange, array, cos, ndarray, sin, stack
from pennylane import draw
from pennylane.tape import QuantumTape

from .. import gates
from ..pauli.sentence import PauliDict
from .evolve import heisenberg
from .pruning import Pruner
from .truncation import FrequencyTruncation, Truncation, WeightTruncation
from .utils import (
    make_evaluator,
    make_sparse_evaluator,
    remove_duplicate_observables,
    requires_propagation,
)

#: Evaluator backends accepted by :meth:`Propagator.propagate`. See that
#: method's docstring for what each one trades off.
_BACKENDS = ("standard", "sparse", "vmap")


def _available_cpus() -> int:
    """
    Number of CPUs actually usable by this process, not the physical machine.

    ``os.cpu_count()`` reports every core on the node, regardless of any
    cgroup/SLURM allocation - on a shared cluster where a job is only granted
    a fraction of a node's cores (e.g. via ``sbatch --cpus-per-task``),
    trusting ``os.cpu_count()`` for ``num_jobs=-1``/``eval_n_jobs=-1`` would
    oversubscribe far past what's actually reserved, hurting this job and
    whatever else is scheduled on the same node. ``os.sched_getaffinity(0)``
    respects that allocation on Linux; fall back to ``os.cpu_count()`` where
    ``sched_getaffinity`` doesn't exist (e.g. macOS).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _propagate_one(paulidict, gates, debug, pruners, truncations):
    """Module-level wrapper around heisenberg() so it can be pickled and
    sent to worker processes. multiprocessing can't pickle bound methods
    or closures reliably, so this has to live at module scope.
    """
    return heisenberg(gates, paulidict, debug, pruners, truncations)

class Propagator:
    """
    Captures and manages a quantum ansatz for symbolic pauli propagation.

    This class records a PennyLane ansatz onto a :class:`~pennylane.tape.QuantumTape`,
    converts its gates to internal :mod:`pprop.gates` representations, and exposes
    methods to propagate observables backwards through the circuit via the Heisenberg
    picture, then evaluate expectations and gradients.

    Parameters
    ----------
    ansatz : Callable
        A PennyLane circuit function that accepts a 1-D array of parameter indices
        and applies quantum operations, returning a list of observables.
    k1 : int, optional
        Pauli weight cutoff. Terms whose Pauli weight exceeds this value are
        discarded during propagation. ``None`` disables truncation.
    k2 : int, optional
        Frequency cutoff. Trigonometric terms whose combined frequency exceeds
        this value are discarded during propagation. ``None`` disables truncation.

    Attributes
    ----------
    ansatz : Callable
        The original ansatz function provided at initialisation.
    tape : pennylane.tape.QuantumTape
        PennyLane quantum tape that records all operations and observables of the
        ansatz.
    observables : list[pennylane.operation.Observable]
        Deduplicated list of observables from the tape.
    paulidicts : list[pprop.pauli.sentence.PauliDict]
        :class:`~pprop.pauli.sentence.PauliDict` representation of each observable,
        used as the starting point for Heisenberg propagation.
    gates : list[pprop.gates.Gate]
        Ordered list of internal :mod:`pprop.gates` gate objects constructed from
        the tape operations. Unrecognised operations are skipped with a warning.
    num_qubits : int
        Number of qubits used by the ansatz, inferred from the tape wires.
    num_params : int
        Number of trainable parameters, inferred as ``max(parameter_indices) + 1``.
    k1 : int or None
        Pauli weight cutoff passed to the propagation routine.
    k2 : int or None
        Frequency cutoff passed to the propagation routine.
    exprs : list[list[tuple[float, list[int], list[int]]]]
        Populated by :meth:`propagate`. Each entry is a list of
        ``(coeff, sin_indices, cos_indices)`` tuples that together encode the
        symbolic expectation value for the corresponding observable.
    _eval_list : list[Callable]
        Populated by :meth:`propagate` (``backend in ("standard", "sparse")``
        only - empty for ``"vmap"``). Fast numeric evaluators
        ``f(sins, coss) -> float`` for each observable, where ``sins`` and
        ``coss`` are ``sin(theta)``/``cos(theta)`` - computed once per
        :meth:`__call__` and shared across every observable rather than each
        one recomputing them from ``theta``.
    _eval_and_grad_list : list[Callable]
        Populated by :meth:`propagate` (``backend in ("standard", "sparse")``
        only - empty for ``"vmap"``). Fast numeric evaluators
        ``f(sins, coss) -> (float, ndarray)`` returning value and gradient for
        each observable, with ``sins``/``coss`` as above.
    _propagated : bool
        Internal flag; ``True`` after :meth:`propagate` has been called
        successfully. Guards methods decorated with :func:`~.utils.requires_propagation`.
    backend : str
        Populated by :meth:`propagate`. One of ``"standard"``, ``"sparse"``,
        ``"vmap"`` - which evaluator implementation ``__call__`` and
        :meth:`eval_and_grad` use. See :meth:`propagate`'s docstring for the
        trade-offs.
    device : str or None
        Populated by :meth:`propagate`. Only meaningful for
        ``backend="vmap"`` - ``None`` unless explicitly requested. See
        :meth:`propagate`'s ``device`` parameter.
    eval_n_jobs : int
        Populated by :meth:`propagate`. Number of threads used to evaluate
        observables in parallel (``backend in ("standard", "sparse")`` only).
        Leave at ``1`` (the default) unless you've measured otherwise for
        your workload - see :meth:`propagate`'s docstring; this was measured
        to hurt, not help, at typical k1/k2-truncated term counts.

    Examples
    --------
    >>> from pprop import Propagator
    >>> import pennylane as qml
    >>> def ansatz(params):
    ...     qml.RX(params[0], wires=0)
    ...     qml.RX(params[1], wires=1)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.RY(params[2], wires=0)
    ...     qml.RY(params[3], wires=1)
    ...     return [qml.expval(qml.PauliZ(0))]
    >>> prop = Propagator(ansatz, k1=None, k2=None)
    >>> print(prop)
    Propagator
      Number of qubits : 2
      Trainable parameters : 4
    """

    def __init__(
        self,
        ansatz: Callable,
        k1: Optional[int] = None,
        k2: Optional[int] = None,
    ):
        # Store user-supplied parameters
        self.k1 = k1
        self.k2 = k2
        self.ansatz = ansatz

        # Capture the ansatz in a quantum tape
        # Integer indices (0 ... 99999) act as a place holder parameter
        # value so we can later read back which parameter slot each gate
        # uses
        with QuantumTape() as self.tape:
            ansatz(arange(100000))

        # Remove duplicate observables
        self.observables, removed_elements = remove_duplicate_observables(self.tape.observables)
        if removed_elements:
            print(f"Removed {len(removed_elements)} duplicate observables")

        # Convert each observable to its PauliDict representation, which is
        # the internal format used during Heisenberg propagation.
        self.paulidicts : List[PauliDict] = [PauliDict.from_qml(observable) for observable in self.observables]

        self.gates : List[gates.Gate] = []
        for op in self.tape.operations:
            if op.name in gates.__all__:
                # Parametrized gates store the integer index of that parameter.
                # Non parametrized gates (e.g CNOT) have no parameter and will
                # just pass None
                parameter = op.parameters[0] if len(op.parameters) == 1 else None
                gate = getattr(gates, op.name)(op.wires, parameter)
                self.gates.append(gate)
            elif op.name == "Barrier": 
                # Barriers are PennyLane no-ops used only for circuit drawing;
                # they carry no physical meaning and can be safely ignored.
                pass
            else:
                print(f"Unknown gate: {op.name}, skipping, consider changing Ansatz")

        # Store tape operations and qubit count
        self.num_qubits : int = len(self.tape.wires)

        # Determine the number of trainable parameters
        params = [int(op.parameters[0]) for op in self.tape.operations if len(op.parameters) == 1]
        self.num_params : int = max(params) + 1 if params else 0
            
        # Guards __call__ and eval_and_grad until propagate() has been run.
        self._propagated : bool = False
        
    # --------------- -
    # Public methods 
    # --------------- -
    def propagate(
        self,
        debug: bool = False,
        pruners: List[Pruner] = [],
        truncations: List[Truncation] = [],
        num_jobs: int = 1,
        backend: str = "standard",
        eval_n_jobs: int = 1,
        device: Optional[str] = None,
    ):
        """
        Propagate each observable backwards through the circuit (Heisenberg picture).

        Evolves each entry of :attr:`paulidicts` through :attr:`gates` in
        reverse (the Heisenberg picture), accumulating the symbolic
        trigonometric expression for its expectation value into
        :attr:`exprs`, then compiles each expression into fast numeric
        callables (:attr:`_eval_list`/:attr:`_eval_and_grad_list`, or a
        single batched callable for ``backend="vmap"``). Idempotent - calling
        this again after a successful call is a no-op (prints a notice and
        returns).

        Parameters
        ----------
        debug : bool, optional
            Print each :class:`~pprop.pauli.sentence.PauliDict` as it is
            propagated. Defaults to ``False``.
        pruners : list[Pruner], optional
            Extra pruning strategies applied during propagation, in addition
            to (not instead of) the built-in ``k1``/``k2`` truncations set at
            construction time. Defaults to ``[]``.
        truncations : list[Truncation], optional
            Extra truncation strategies applied during propagation, in
            addition to the built-in ``k1``/``k2`` ones. Defaults to ``[]``.
        num_jobs : int, optional
            Number of worker processes used to propagate observables in
            parallel. Each entry in :attr:`paulidicts` is evolved
            independently, so this loop parallelizes cleanly across
            processes. Defaults to ``1`` (sequential — identical behaviour
            and ordering to before). Pass ``-1`` to use all available cores
            (:func:`_available_cpus` - respects a cgroup/SLURM allocation
            rather than the whole node's core count), or any positive
            integer for that many worker processes. Requires :attr:`gates`,
            ``pruners``, and ``truncations`` to be picklable. When
            ``num_jobs > 1`` and ``debug=True``, printed output from
            different observables will be interleaved/out of order since it
            comes from separate processes. Workers use the platform-default
            start method (``"fork"`` on Linux). **Known issue:** combining
            ``num_jobs > 1`` with ``backend="vmap"`` in the same process can
            deadlock - forking after JAX has started its background threads
            can inherit a mutex mid-hold by a thread that no longer exists in
            the child. (``"spawn"`` avoids that specific bug but was measured
            far slower here - every worker re-imports this whole module tree,
            PennyLane and JAX included, from scratch - so it was not adopted;
            see ``tests/test_eval_and_grad_jit_bench.py``/the vmap backend's
            module docstring for the current state of this trade-off.) Until
            resolved, use ``num_jobs=1`` when propagating with
            ``backend="vmap"``.

            Note ``num_jobs`` only affects *propagation* (this one-time setup
            step); it has nothing to do with ``eval_n_jobs`` below, which
            affects every subsequent call to :meth:`__call__`/:meth:`eval_and_grad`.
        backend : {"standard", "sparse", "vmap"}, optional
            Which evaluator implementation to build once propagation
            finishes. All three compute *exactly* the same values and
            gradients (verified in ``tests/test_backends.py``) - this only
            picks the internal representation, purely a speed/engineering
            trade-off. See ``notebooks/test/sparse_arrays_explained.ipynb``
            and ``tests/test_eval_and_grad_jit_bench.py`` for how these were
            benchmarked. Defaults to ``"standard"`` (unchanged behaviour from
            previous versions of this library).

            - ``"standard"``: the original dense ``(n_terms, num_params)``
              arrays (:func:`~.utils.make_evaluator`). Safe default; every
              term carries one column per circuit parameter even if it only
              touches a handful of them.
            - ``"sparse"``: narrower ``(n_terms, W)`` gathered arrays
              (:func:`~.utils.make_sparse_evaluator`), ``W`` = the most
              parameters any single term touches. Same NumPy, same single-core
              execution model as ``"standard"`` - just less wasted work per
              term. Measured ~6x faster than ``"standard"`` at k1/k2
              truncation levels that keep terms narrow (the more aggressive
              your truncation, the bigger the win). **Recommended starting
              point if training is slow.**
            - ``"vmap"``: batches *all* observables into a single
              ``jax.jit`` + ``jax.vmap`` call
              (:func:`~.vmap_backend.make_batched_evaluator`), built on the
              same sparse arrays as ``"sparse"``. On CPU this was measured
              *slower* than ``"sparse"`` in every test so far (JAX's gradient
              scatter-add is disproportionately expensive on CPU) - **on
              GPU it flips**: ~70-80x faster than the same batch on CPU, and
              ~4x faster than ``"sparse"`` on CPU, at the 1000-observable
              scale measured in ``notebooks/test/gpu_backend.ipynb``. That GPU win
              only materialises for a large batch of observables evaluated
              together, though - per-call dispatch/transfer overhead
              dominates for the common case of propagating a handful of
              observables at a time (MMD-style training with hundreds of
              observables per round, as in ``scripts/generation/train.py``,
              is the exception, not the rule) - which is why ``"sparse"``
              remains the default even when a GPU is available. See
              ``device`` below to opt into GPU explicitly. Requires
              `jax`/`jaxlib` (already a project dependency; GPU support
              additionally requires a CUDA-enabled jaxlib/plugin - see
              ``notebooks/test/gpu_backend.ipynb`` for the install command).
              Ignores ``eval_n_jobs`` (there's no per-observable Python loop
              left to parallelize - everything happens in one compiled call).
        eval_n_jobs : int, optional
            Number of threads used to evaluate observables in parallel on
            every call to :meth:`__call__`/:meth:`eval_and_grad`, for
            ``backend in ("standard", "sparse")``. Defaults to ``1``
            (sequential Python loop - this is the recommended value).

            **Measured not to help, and often to hurt**, at the term counts
            k1/k2 truncation actually produces (tens to a few hundred terms
            per observable): each observable's evaluation only takes on the
            order of 0.1-0.2 ms, which is too little NumPy-releases-the-GIL
            work per call to amortize Python's GIL acquire/release and
            thread-scheduling overhead - measured consistently *slower* than
            ``eval_n_jobs=1`` across every thread count from 2 to 128, with
            both this implementation and a from-scratch ``joblib`` version
            (an earlier "measured ~1.4x speedup at 8 threads" claim in this
            docstring did not reproduce under more careful/repeated testing
            and was retracted - that number should not have been trusted).
            Threading only has a chance of paying off if a single
            observable's own NumPy work is heavy enough to dominate GIL
            overhead - e.g. propagating with little/no ``k1``/``k2``
            truncation, so individual terms counts run into the thousands+
            (see :func:`~.utils.make_sparse_evaluator`'s big-``n_terms``
            timings in ``tests/test_eval_and_grad_jit_bench.py``). For a
            typical truncated propagation, leave this at ``1``; for genuine
            multi-core scaling use ``num_jobs`` (separate *processes*, no
            GIL) for the one-time propagation step, or split work across
            independent OS processes/SLURM jobs as ``TRAIN.sh`` already does
            for full training runs.

            Pass ``-1`` to use all available cores (via
            ``os.sched_getaffinity`` where supported, so this respects a
            cgroup/SLURM allocation rather than reporting the whole node's
            core count - see :func:`_available_cpus`; same convention as
            ``num_jobs`` above) if you want to experiment regardless. The
            thread pool is created once here and reused for the life of this
            :class:`Propagator` - ``eval_n_jobs`` cannot be changed without
            calling :meth:`propagate` again (which isn't supported once
            already propagated - build a new :class:`Propagator` instead).
        device : {"cpu", "gpu", "tpu"}, optional
            Only meaningful for ``backend="vmap"`` - raises :exc:`ValueError`
            if set for ``"standard"``/``"sparse"`` (they're plain NumPy;
            there's no device to place them on). Defaults to ``None``, which
            leaves it to JAX's own default device selection (GPU
            automatically, if a CUDA-enabled jaxlib/plugin is installed).
            Pass ``"gpu"`` to opt in explicitly - worthwhile when you're
            batching many observables together (e.g. MMD-style training),
            per ``notebooks/test/gpu_backend.ipynb``'s measurements - or
            ``"cpu"`` to force CPU even if a GPU is present (e.g. to avoid
            contending for a shared GPU for a workload too small to
            benefit). Raises :exc:`ValueError` if the requested backend
            isn't available in this environment.
        """
        if self._propagated:
            print("Already propagated")
            return

        if backend not in _BACKENDS:
            raise ValueError(f"backend must be one of {_BACKENDS}, got {backend!r}")
        if device is not None and backend != "vmap":
            raise ValueError(
                f"device={device!r} is only meaningful for backend='vmap' "
                f"(got backend={backend!r}) - the other backends are plain "
                "NumPy, there's no device to place them on."
            )
        if eval_n_jobs == -1:
            eval_n_jobs = _available_cpus()
        elif eval_n_jobs < 1:
            raise ValueError(f"eval_n_jobs must be -1 or a positive integer, got {eval_n_jobs}")
        if backend == "vmap" and eval_n_jobs > 1:
            print('backend="vmap" ignores eval_n_jobs (there is no per-observable '
                  'Python loop left to thread - everything runs in one batched call).')

        builtin_truncations: List[Truncation] = []
        if self.k1 is not None:
            builtin_truncations.append(WeightTruncation(self.k1))
        if self.k2 is not None:
            builtin_truncations.append(FrequencyTruncation(self.k2))
        all_truncations = builtin_truncations + list(truncations)

        self.exprs = [None] * len(self.paulidicts)
        self.history = [None] * len(self.paulidicts)

        if num_jobs == -1:
            num_jobs = _available_cpus()
        elif num_jobs < 1:
            raise ValueError(f"num_jobs must be -1 or a positive integer, got {num_jobs}")

        if num_jobs == 1:
            for i, paulidict in enumerate(self.paulidicts):
                if debug:
                    print("Propagating", paulidict)
                propagationdicts, propagationexprs, history = heisenberg(
                    self.gates, paulidict, debug, pruners, all_truncations
                )
                self.paulidicts[i], self.exprs[i] = propagationdicts, propagationexprs
                self.history[i] = history
        else:
            worker = partial(
                _propagate_one,
                gates=self.gates,
                debug=debug,
                pruners=pruners,
                truncations=all_truncations,
            )
            # Platform-default start method ("fork" on Linux): fast, and
            # fine for the common case (no already-multithreaded library in
            # this process). NOTE this can deadlock if combined with
            # backend="vmap" - forking after JAX has started background
            # threads can inherit a mutex mid-hold by a thread that no
            # longer exists in the child. Tried switching to "spawn" to
            # avoid that class of bug; on this codebase's import-heavy
            # module tree (PennyLane, JAX, ...) that made every num_jobs>1
            # run - not just backend="vmap" - far slower or apparently hang
            # (each worker re-imports everything from scratch), so it was
            # reverted. If you hit the vmap+num_jobs>1 deadlock, the
            # practical workaround today is num_jobs=1 (propagate
            # sequentially) when using backend="vmap".
            with get_context("fork").Pool(num_jobs) as pool:
                results = pool.map(worker, self.paulidicts)

            for i, (propagationdicts, propagationexprs, history) in enumerate(results):
                self.paulidicts[i], self.exprs[i] = propagationdicts, propagationexprs
                self.history[i] = history

        self.backend = backend
        self.device = device
        self.eval_n_jobs = eval_n_jobs
        self._executor: Optional[ThreadPoolExecutor] = None
        self._batched_eval: Optional[Callable] = None
        self._batched_eval_and_grad: Optional[Callable] = None
        self._raw_eval_and_grad: Optional[Callable] = None

        if backend == "vmap":
            from .vmap_backend import make_batched_evaluator  # local: defers jax import/config

            self._batched_eval, self._batched_eval_and_grad, self._raw_eval_and_grad = (
                make_batched_evaluator(self.exprs, self.num_params, device=device)
            )
            # Not used by this backend, but kept defined (as empty lists) so any
            # external code introspecting these attributes doesn't hit AttributeError.
            self._eval_list = []
            self._eval_and_grad_list = []
        else:
            build_fn = make_evaluator if backend == "standard" else make_sparse_evaluator
            self._eval_list = []
            self._eval_and_grad_list = []
            for expr in self.exprs:
                fg = build_fn(expr, self.num_params)
                self._eval_list.append(fg[0])
                self._eval_and_grad_list.append(fg[1])

            if eval_n_jobs > 1:
                self._executor = ThreadPoolExecutor(max_workers=eval_n_jobs)

        self._propagated = True
        
    def show(self) -> None:
        """
        Print an ASCII drawing of the quantum circuit to stdout.

        Uses PennyLane's :func:`~pennylane.draw` utility with the integer
        parameter indices ``0 … num_params-1`` as placeholder values.
        """
        drawer = draw(self.ansatz)
        print(drawer(arange(self.num_params)))

    def expression(self, idx: int = 0):
        """
        Reconstruct the SymPy expectation-value expression for a given observable.

        Converts the compact ``(coeff, sin_indices, cos_indices)`` tuples stored
        in :attr:`exprs` back into a human-readable :class:`sympy.Expr` in terms
        of symbolic angles ``θ0, θ1, …``.

        Parameters
        ----------
        idx : int, optional
            Index into :attr:`exprs` selecting which observable to reconstruct.
            Defaults to ``0`` (the first observable).

        Returns
        -------
        sympy.Expr
            The full symbolic expression for the expectation value.
            Returns ``sympy.S.Zero`` if the expression list for ``idx`` is empty.

        Raises
        ------
        IndexError
            If ``idx`` is out of range for :attr:`exprs`.
        """
        from sympy import Add, Mul, S, cos, sin, symbols

        expr = self.exprs[idx]

        # An empty expression list means the observable evaluates to zero.
        if not expr:
            return S.Zero

        # Create real symbolic angles θ0, θ1, …, θ_{num_params-1}.
        theta = symbols(f"θ0:{self.num_params}", real=True)

        terms = []
        for coeff, sin_idx, cos_idx in expr:
            # Each term is a product of a numeric coefficient with zero or more
            # sin/cos factors, one per parameter index in sin_idx / cos_idx.
            factors = [coeff]
            for i in sin_idx:
                factors.append(sin(theta[i]))
            for i in cos_idx:
                factors.append(cos(theta[i]))
            terms.append(Mul(*factors))

        return Add(*terms)

    # --------------- -
    # Dunder methods 
    # --------------- -

    def __repr__(self) -> str:
        """
        Return a concise human-readable summary of the propagator.

        Returns
        -------
        str
            Multi-line string listing the number of qubits and trainable parameters.
        """
        reprstr = "Propagator\n"
        reprstr += f"  Number of qubits : {self.num_qubits}\n"
        reprstr += f"  Trainable parameters : {self.num_params}\n"
        return reprstr

    @requires_propagation
    def __call__(self, params: ndarray) -> ndarray:
        """
        Evaluate all observable expectation values at the given parameters.

        Requires :meth:`propagate` to have been called first. Dispatches to
        whichever ``backend`` was chosen in :meth:`propagate` - the result is
        identical either way, only the internal computation differs.

        Parameters
        ----------
        params : ndarray of shape (num_params,)
            Numeric values for the circuit's trainable parameters.

        Returns
        -------
        ndarray of shape (num_observables,)
            Expectation value of each observable at ``params``.
        """
        if self.backend == "vmap":
            return self._batched_eval(params)
        sins, coss = sin(params), cos(params)
        if self._executor is not None:
            return array(list(self._executor.map(lambda f: f(sins, coss), self._eval_list)))
        return array([f(sins, coss) for f in self._eval_list])

    @requires_propagation
    def eval_and_grad(self, params: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluate expectation values and their parameter gradients simultaneously.

        Requires :meth:`propagate` to have been called first. Dispatches to
        whichever ``backend`` was chosen in :meth:`propagate` - the result is
        identical either way, only the internal computation differs.

        Parameters
        ----------
        params : ndarray of shape (num_params,)
            Numeric values for the circuit's trainable parameters.

        Returns
        -------
        vals : ndarray of shape (num_observables,)
            Expectation value of each observable at ``params``.
        grads : ndarray of shape (num_observables, num_params)
            Gradient of each expectation value with respect to each parameter.
        """
        if self.backend == "vmap":
            return self._batched_eval_and_grad(params)

        sins, coss = sin(params), cos(params)
        if self._executor is not None:
            results = list(self._executor.map(lambda f: f(sins, coss), self._eval_and_grad_list))
        else:
            results = [f(sins, coss) for f in self._eval_and_grad_list]

        # Unzip the list of (value, gradient) pairs into two separate arrays.
        vals  = array([v for v, _ in results])   # shape: (num_observables,)
        grads = stack([g for _, g in results])    # shape: (num_observables, num_params)

        return vals, grads
