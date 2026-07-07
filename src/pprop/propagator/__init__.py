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
from functools import partial
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple

from numpy import arange, array, ndarray, stack
from pennylane import draw
from pennylane.tape import QuantumTape

from .. import gates
from ..pauli.sentence import PauliDict
from .evolve import heisenberg
from .pruning import Pruner
from .truncation import FrequencyTruncation, Truncation, WeightTruncation
from .utils import make_evaluator, remove_duplicate_observables, requires_propagation


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
        Populated by :meth:`propagate`. Fast numeric evaluators
        ``f(params) -> float`` for each observable.
    _eval_and_grad_list : list[Callable]
        Populated by :meth:`propagate`. Fast numeric evaluators
        ``f(params) -> (float, ndarray)`` returning value and gradient for each
        observable.
    _propagated : bool
        Internal flag; ``True`` after :meth:`propagate` has been called
        successfully. Guards methods decorated with :func:`~.utils.requires_propagation`.

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
    ):
        """
        Propagate each observable backwards through the circuit (Heisenberg picture).

        ... (existing docstring text unchanged) ...

        num_jobs : int, optional
            Number of worker processes used to propagate observables in
            parallel. Each entry in :attr:`paulidicts` is evolved
            independently, so this loop parallelizes cleanly across
            processes. Defaults to ``1`` (sequential — identical behaviour
            and ordering to before). Requires :attr:`gates`, ``pruners``,
            and ``truncations`` to be picklable. When ``num_jobs > 1`` and
            ``debug=True``, printed output from different observables will
            be interleaved/out of order since it comes from separate
            processes.
        """
        if self._propagated:
            print("Already propagated")
            return

        builtin_truncations: List[Truncation] = []
        if self.k1 is not None:
            builtin_truncations.append(WeightTruncation(self.k1))
        if self.k2 is not None:
            builtin_truncations.append(FrequencyTruncation(self.k2))
        all_truncations = builtin_truncations + list(truncations)

        self.exprs = [None] * len(self.paulidicts)
        self.history = [None] * len(self.paulidicts)

        if num_jobs == -1:
            num_jobs = os.cpu_count()
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
            with Pool(num_jobs) as pool:
                results = pool.map(worker, self.paulidicts)

            for i, (propagationdicts, propagationexprs, history) in enumerate(results):
                self.paulidicts[i], self.exprs[i] = propagationdicts, propagationexprs
                self.history[i] = history

        self._eval_list: List[Callable] = []
        self._eval_and_grad_list: List[Callable] = []

        for expr in self.exprs:
            fg = make_evaluator(expr, self.num_params)
            self._eval_list.append(fg[0])
            self._eval_and_grad_list.append(fg[1])

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
        return array([f(params) for f in self._eval_list])

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
        results = [f(params) for f in self._eval_and_grad_list]

        # Unzip the list of (value, gradient) pairs into two separate arrays.
        vals  = array([v for v, _ in results])   # shape: (num_observables,)
        grads = stack([g for _, g in results])    # shape: (num_observables, num_params)

        return vals, grads
