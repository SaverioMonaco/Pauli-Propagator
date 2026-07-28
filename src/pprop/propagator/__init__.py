"""
Core module with the Propagator.
Propagator takes as an input a quantum circuit as a function of a list of parameters List[float] and returns
the expectation value of an observable.

Propagation itself (the Heisenberg-picture evolution of each observable
backward through the circuit's gates) runs entirely in the Rust extension
``pprop_rs``. This fork of pprop has no pure-Python propagation path anymore.
See ``native/pprop_rs`` and ``personal/rust_port.tex`` for why, and for measured
performance numbers versus the pure-Python implementation this replaced.

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
from typing import Callable, List, Optional, Sequence, Tuple, Union

from numpy import arange, array, cos, empty, integer, ndarray, sin
from pennylane import draw
from pennylane.tape import QuantumTape

import pprop_rs

from .. import gates
from ..pauli.sentence import PauliDict
from .binding import BoundPropagator, Free, affine_from_exprs
from .utils import make_sparse_evaluator, remove_duplicate_observables, requires_propagation

#: Gate class name (pennylane's Operator.name) -> pprop_rs gate-kind code.
#: Must stay in sync with the kind constants at the top of
#: native/pprop_rs/src/lib.rs.
_GATE_KIND = {
    "RX": 0, "RY": 1, "RZ": 2,
    "Hadamard": 3, "S": 4, "SX": 5, "T": 6,
    "SWAP": 7, "CNOT": 8, "CY": 9, "CZ": 10,
    "CRX": 11, "CRY": 12, "CRZ": 13,
}


def _words_needed(num_qubits: int) -> int:
    """
    Smallest power-of-two count of 64-bit words covering ``num_qubits``.

    Must match ``words_needed()`` in ``native/pprop_rs/src/lib.rs`` exactly -
    it picks which of that file's const-generic ``NW`` monomorphizations
    handles this circuit, and this function picks how many ``u64`` limbs
    :func:`_int_to_words` chunks each :class:`~pprop.pauli.op.PauliOp` into
    for the trip across that boundary. A mismatch would silently truncate
    high qubit indices.
    """
    raw = max(1, -(-num_qubits // 64))  # ceil(num_qubits / 64), at least 1
    return 1 << (raw - 1).bit_length()


def _int_to_words(value: int, n_words: int) -> list[int]:
    """
    Chunk a non-negative arbitrary-precision int into ``n_words`` little-endian u64 limbs.

    :class:`~pprop.pauli.op.PauliOp` stores ``x``/``z`` as plain Python ints
    (no qubit-count limit); this is the only place that width gets folded
    down to the fixed-size ``[u64; NW]`` arrays ``pprop_rs`` operates on.
    """
    mask = (1 << 64) - 1
    return [(value >> (64 * i)) & mask for i in range(n_words)]


def _available_cpus() -> int:
    """
    Number of CPUs actually usable by this process, not the physical machine.

    ``os.cpu_count()`` reports every core on the node, regardless of any
    cgroup/SLURM allocation - on a shared cluster where a job is only granted
    a fraction of a node's cores (e.g. via ``sbatch --cpus-per-task``),
    trusting ``os.cpu_count()`` for ``eval_n_jobs=-1`` would oversubscribe
    far past what's actually reserved, hurting this job and whatever else is
    scheduled on the same node. ``os.sched_getaffinity(0)`` respects that
    allocation on Linux; fall back to ``os.cpu_count()`` where
    ``sched_getaffinity`` doesn't exist (e.g. macOS).
    """
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Propagator:
    """
    Captures and manages a quantum ansatz for symbolic pauli propagation.

    This class records a PennyLane ansatz onto a :class:`~pennylane.tape.QuantumTape`,
    converts its gates to internal :mod:`pprop.gates` representations, and exposes
    methods to propagate observables backwards through the circuit via the Heisenberg
    picture, then evaluate expectations and gradients.

    Circuits are still defined with PennyLane, exactly as before. Only the
    propagation step itself (:meth:`propagate`) has moved to Rust.

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
        Number of trainable parameters, inferred as ``max(parameter_indices) + 1``
        over gates given an int/``np.integer`` parameter. Gates given a plain
        float parameter (e.g. ``qml.RY(0.3, wires=q)``) are fixed, non-trainable
        values and don't count towards this: see :attr:`_fixed_value_slots`.
    k1 : int or None
        Pauli weight cutoff passed to the propagation routine.
    k2 : int or None
        Frequency cutoff passed to the propagation routine.
    exprs : list[list[tuple[float, list[int], list[int]]]]
        Populated by :meth:`propagate`. Each entry is a list of
        ``(coeff, sin_indices, cos_indices)`` tuples that together encode the
        symbolic expectation value for the corresponding observable.
    _eval_list : list[Callable]
        Populated by :meth:`propagate`. Fast numeric evaluators
        ``f(sins, coss) -> float`` for each observable, where ``sins`` and
        ``coss`` are ``sin(theta)``/``cos(theta)`` - computed once per
        :meth:`__call__` and shared across every observable rather than each
        one recomputing them from ``theta``.
    _eval_and_grad_list : list[Callable]
        Populated by :meth:`propagate`. Fast numeric evaluators
        ``f(sins, coss) -> (float, ndarray)`` returning value and gradient for
        each observable, with ``sins``/``coss`` as above.
    _fixed_value_slots : dict[float, int]
        Maps each distinct fixed gate value to a hidden slot index right
        after the trainable range ``[0, num_params)``. Populated once in
        :meth:`__init__`; empty (and free of overhead) for circuits with no
        fixed-value gates.
    _internal_num_params : int
        ``num_params`` plus the number of distinct fixed values, i.e. the
        width of the padded array :meth:`_full_params` builds before every
        ``sin``/``cos`` evaluation.
    _propagated : bool
        Internal flag; ``True`` after :meth:`propagate` has been called
        successfully. Guards methods decorated with :func:`~.utils.requires_propagation`.
    eval_n_jobs : int
        Populated by :meth:`propagate`. Number of threads used to evaluate
        observables in parallel. Leave at ``1`` (the default) unless you've
        measured otherwise for your workload: each observable's evaluation
        is typically too cheap (~0.1-0.2ms) for threading to pay off.

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

        # Determine the number of trainable parameters. Only int/np.integer
        # parameters are real trainable indices (see Gate's docstring); a
        # float parameter (e.g. qml.RY(0.3, wires=q), or any arithmetic on
        # the placeholder that promotes it to float) is a *fixed*, non
        # trainable value and must not be counted or truncated into an index.
        index_params = [
            int(g.parameter) for g in self.gates
            if g.parameter is not None and isinstance(g.parameter, (int, integer))
        ]
        self.num_params : int = max(index_params) + 1 if index_params else 0

        # Fixed-value gates don't get a user-facing slot in `num_params`;
        # instead each *distinct* fixed value is assigned its own hidden slot
        # right after the real trainable indices, so it can still flow through
        # the same sin(theta)/cos(theta) machinery as everything else.
        # `_full_params` splices these constants in before every evaluation.
        self._fixed_value_slots : dict = {}
        next_slot = self.num_params
        for g in self.gates:
            if g.parameter is not None and not isinstance(g.parameter, (int, integer)):
                value = float(g.parameter)
                if value not in self._fixed_value_slots:
                    self._fixed_value_slots[value] = next_slot
                    next_slot += 1
        self._internal_num_params : int = next_slot

        # Guards __call__ and eval_and_grad until propagate() has been run.
        self._propagated : bool = False

    # --------------- -
    # Public methods
    # --------------- -
    def propagate(
        self,
        use_dead_qubit_pruner: bool = False,
        use_xy_weight_pruner: bool = False,
        coeff_threshold: Optional[float] = None,
        eval_n_jobs: int = 1,
    ):
        """
        Propagate each observable backwards through the circuit (Heisenberg picture).

        Evolves every entry of :attr:`paulidicts` through :attr:`gates` in
        reverse (the Heisenberg picture) inside the Rust extension
        ``pprop_rs``, accumulating each observable's symbolic
        trigonometric expectation-value expression into :attr:`exprs`, then
        compiles each expression into a fast numeric callable
        (:attr:`_eval_list`/:attr:`_eval_and_grad_list`, via the "sparse"
        gathered-array representation; see ``utils.make_sparse_evaluator``).
        Idempotent: calling this again after a successful call is a no-op
        (prints a notice and returns).

        Parameters
        ----------
        use_dead_qubit_pruner : bool, optional
            Enable ``DeadQubitPruner``: exact pruning of words carrying a
            frozen X/Y on a qubit no remaining gate can ever touch again.
            Defaults to ``False`` (matches the previous default of not
            passing any pruners).
        use_xy_weight_pruner : bool, optional
            Enable ``XYWeightPruner``: exact pruning of words whose XY-weight
            exceeds the maximum reduction achievable by the remaining
            circuit. Defaults to ``False``.
        coeff_threshold : float, optional
            If given, any :data:`~pprop.pauli.sentence.CoeffTerm` whose
            scalar magnitude falls below this threshold is discarded after
            every gate step. This is approximate, equivalent to the old
            ``CoefficientTruncation``. ``None`` (default) disables this.
        eval_n_jobs : int, optional
            Number of threads used to evaluate observables in parallel on
            every call to :meth:`__call__`/:meth:`eval_and_grad`. Defaults
            to ``1`` (sequential Python loop, which is recommended: each
            observable's evaluation is typically too cheap for threading to
            pay off). Pass ``-1`` to use all available cores (respecting a
            cgroup/SLURM allocation via ``os.sched_getaffinity``).
        """
        if self._propagated:
            print("Already propagated")
            return

        if eval_n_jobs == -1:
            eval_n_jobs = _available_cpus()
        elif eval_n_jobs < 1:
            raise ValueError(f"eval_n_jobs must be -1 or a positive integer, got {eval_n_jobs}")

        gate_kind, gate_wire0, gate_wire1, gate_param = [], [], [], []
        for g in self.gates:
            name = g.qml_gate.name
            if name not in _GATE_KIND:
                raise ValueError(
                    f"Gate {name!r} has no Rust propagation rule in pprop_rs "
                    f"(supported: {sorted(_GATE_KIND)})."
                )
            gate_kind.append(_GATE_KIND[name])
            gate_wire0.append(int(g.wires[0]))
            gate_wire1.append(int(g.wires[1]) if len(g.wires) > 1 else -1)
            if g.parameter is None:
                gate_param.append(-1)
            elif isinstance(g.parameter, (int, integer)):
                gate_param.append(int(g.parameter))
            else:
                # Fixed-value gate: resolve to its hidden slot (see __init__),
                # not to int(g.parameter), which would silently truncate the
                # value and alias it onto an unrelated trainable index.
                gate_param.append(self._fixed_value_slots[float(g.parameter)])

        # pprop_rs packs each Pauli word's x/z plane into a handful of u64
        # words (see native/pprop_rs/src/lib.rs) rather than one u64, so it
        # isn't limited to 64 qubits the way a single bitmask would be.
        # PauliOp.x/.z stay plain (arbitrary-precision) Python ints; only at
        # this boundary do we chunk them into little-endian u64 limbs.
        n_words = _words_needed(self.num_qubits)

        rust_paulidicts = []
        for paulidict in self.paulidicts:
            spec = []
            for op, terms in paulidict.items():
                x_words = _int_to_words(op.x, n_words)
                z_words = _int_to_words(op.z, n_words)
                for (c, s, cc) in terms:
                    spec.append((x_words, z_words, float(c), list(s), list(cc)))
            rust_paulidicts.append(spec)

        self.exprs = pprop_rs.propagate_batch(
            self.num_qubits,
            gate_kind, gate_wire0, gate_wire1, gate_param,
            self.k1 if self.k1 is not None else -1,
            self.k2 if self.k2 is not None else -1,
            coeff_threshold if coeff_threshold is not None else -1.0,
            use_dead_qubit_pruner, use_xy_weight_pruner,
            rust_paulidicts,
        )

        self.eval_n_jobs = eval_n_jobs
        self._executor: Optional[ThreadPoolExecutor] = None

        self._eval_list = []
        self._eval_and_grad_list = []
        for expr in self.exprs:
            fg = make_sparse_evaluator(expr, self._internal_num_params)
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

        # Real trainable indices get a symbolic angle θ0, θ1, …; fixed-value
        # gates' hidden slots (see __init__) get their literal numeric value
        # instead, since they aren't free variables of this expression.
        theta = list(symbols(f"θ0:{self.num_params}", real=True))
        value_by_slot = {slot: value for value, slot in self._fixed_value_slots.items()}
        theta += [value_by_slot[i] for i in range(self.num_params, self._internal_num_params)]

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

    def bind(self, exprs: Sequence[Union[Free, float]]) -> BoundPropagator:
        """
        Reparametrise this propagator's ``num_params``-sized parameter vector
        as an affine function of a smaller, user-defined "free" vector, e.g.
        one gate reading ``f0`` and another reading ``-2 * f0``.

        This needs no support from :meth:`propagate` itself: pprop's own
        gradient is already exact closed-form calculus, so any affine
        dependency between gate angles is handled by a single Jacobian
        multiply, exactly (not approximately), via the chain rule.

        Parameters
        ----------
        exprs : sequence of Free or float
            One entry per index in ``range(num_params)`` (in the same order
            gates were captured from the ansatz), each either a :class:`Free`
            expression built from :meth:`Free.vars` via ``+``, ``-``, ``*``,
            ``/`` with plain numbers, or a plain number for a fixed value at
            that index.

        Returns
        -------
        BoundPropagator
            Callable wrapper exposing ``__call__(free)`` and
            ``eval_and_grad(free)`` over the free-parameter vector.

        Examples
        --------
        >>> def ansatz(params):
        ...     qml.RY(params[0], wires=0)   # will represent: f0
        ...     qml.RX(params[1], wires=1)   # will represent: -2 * f0
        ...     qml.RZ(params[2], wires=1)   # will represent: f1
        >>> prop = Propagator(ansatz)
        >>> prop.propagate()
        >>> f0, f1 = Free.vars(2)
        >>> bound = prop.bind([f0, -2 * f0, f1])
        >>> vals, grad = bound.eval_and_grad(np.array([0.3, 1.1]))
        """
        J, b, _ = affine_from_exprs(exprs, self.num_params)
        return BoundPropagator(self, J, b)

    def _full_params(self, params: ndarray) -> ndarray:
        """
        Pad ``params`` (length :attr:`num_params`) with fixed-value gates'
        hidden slots (length :attr:`_internal_num_params`), so ``sin``/``cos``
        can be computed once over the full internal parameter vector.

        Passthrough (no allocation) when there are no fixed-value gates.
        """
        if not self._fixed_value_slots:
            return params
        full = empty(self._internal_num_params)
        full[: self.num_params] = params
        for value, slot in self._fixed_value_slots.items():
            full[slot] = value
        return full

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

        Requires :meth:`propagate` to have been called first.

        Parameters
        ----------
        params : ndarray of shape (num_params,)
            Numeric values for the circuit's trainable parameters.

        Returns
        -------
        ndarray of shape (num_observables,)
            Expectation value of each observable at ``params``.
        """
        full = self._full_params(params)
        sins, coss = sin(full), cos(full)
        if self._executor is not None:
            return array(list(self._executor.map(lambda f: f(sins, coss), self._eval_list)))
        return array([f(sins, coss) for f in self._eval_list])

    @requires_propagation
    def eval_and_grad(self, params: ndarray) -> Tuple[ndarray, ndarray]:
        """
        Evaluate expectation values and their parameter gradients simultaneously.

        Requires :meth:`propagate` to have been called first.

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
        from numpy import stack

        full = self._full_params(params)
        sins, coss = sin(full), cos(full)
        if self._executor is not None:
            results = list(self._executor.map(lambda f: f(sins, coss), self._eval_and_grad_list))
        else:
            results = [f(sins, coss) for f in self._eval_and_grad_list]

        # Unzip the list of (value, gradient) pairs into two separate arrays.
        vals  = array([v for v, _ in results])   # shape: (num_observables,)
        grads = stack([g for _, g in results])    # shape: (num_observables, _internal_num_params)

        # Fixed-value slots aren't trainable, drop their gradient columns so
        # callers only ever see one column per entry of the params they passed in.
        return vals, grads[:, : self.num_params]
