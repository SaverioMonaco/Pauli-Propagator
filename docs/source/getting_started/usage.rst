Usage
-----

Pauli propagation works in the Heisenberg picture: rather than simulating the
statevector, it evolves each observable *backwards* through the circuit gate by
gate, producing a closed-form trigonometric polynomial in the parameters. The
workflow has four steps.

Step 0: Imports
~~~~~~~~~~~~~~~~

.. code-block:: python

    import pennylane as qml       # To define the circuit
    from pprop import Propagator  # For Pauli Propagation

Step 1: Define the ansatz
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ansatz must be a plain Python function that accepts a parameter array,
applies PennyLane gates, and returns a **list** of ``qml.expval(...)`` calls,
one per observable. Observables can be single Pauli words or arbitrary linear
combinations thereof.

.. note::
    ``qml.Barrier()`` is a PennyLane no-op used only for circuit drawing.
    It is automatically ignored by the propagator.

.. code-block:: python

    def ansatz(params: list[float]):
        qml.RX(params[0], wires=0)
        qml.RX(params[1], wires=1)
        qml.RY(params[2], wires=0)
        qml.RY(params[3], wires=1)
        qml.Hadamard(wires=2)
        qml.Barrier()
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.Barrier()
        qml.RY(params[4], wires=0)
        qml.RY(params[5], wires=1)
        qml.RY(params[6], wires=2)
        return [
            qml.expval(qml.PauliZ(0)),                                  # ⟨Z₀⟩
            qml.expval(qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2)),  # ⟨X₀X₁X₂⟩
            qml.expval(qml.PauliY(2)),                                  # ⟨Y₂⟩
            qml.expval(-qml.PauliX(0) @ qml.PauliX(1) @ qml.PauliX(2)   
                       + 13 * qml.PauliZ(2)),                           # ⟨-X₀X₁X₂ + 13Z₂⟩
        ]

Step 2: Create the Propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Propagator`` wraps the ansatz and accepts two optional truncation cutoffs:

- ``k1``: **Pauli weight cutoff**, discard evolved Pauli words with more than
  ``k1`` non-identity single-qubit factors. Useful for large systems where
  high-weight terms contribute negligibly.
- ``k2``: **Frequency cutoff**, discard terms whose total number of
  trigonometric factors exceeds ``k2``. Controls expression complexity.

Setting both to ``None`` performs exact propagation with no approximation.

.. code-block:: python

    prop = Propagator(
        ansatz,
        k1=None,  # Pauli weight cutoff  (None = exact)
        k2=None,  # Frequency cutoff     (None = exact)
    )

.. code-block:: text

    >>> prop
    Propagator
      Number of qubits : 3
      Trainable parameters : 7

Step 3: Propagate
~~~~~~~~~~~~~~~~~~

``.propagate()`` evolves each observable backwards through the circuit, via
the Rust extension ``pprop_rs`` (see :doc:`/native_pprop_rs`). It runs
silently and returns nothing; the propagated expressions end up in
``prop.exprs``.

.. code-block:: python

    prop.propagate()

Two optional exact pruning heuristics can be enabled on top of the
``k1``/``k2`` cutoffs, which discard terms that provably cannot contribute
to the final expectation value:

.. code-block:: python

    prop.propagate(use_dead_qubit_pruner=True, use_xy_weight_pruner=True)

Step 4: Evaluate
~~~~~~~~~~~~~~~~~

Once propagated, calling ``prop(params)`` returns all expectation values as a
NumPy array of shape ``(num_observables,)``.

.. code-block:: python

    random_params = qml.numpy.arange(prop.num_params)
    prop_output = prop(random_params)

.. code-block:: text

    >>> prop_output
    [ 0.32448207 -0.5280619   0.          4.16046337]

Gradients are available via ``.eval_and_grad()``, which returns
``(values, jacobian)`` with shapes ``(num_obs,)`` and ``(num_obs, num_params)``
respectively:

.. code-block:: python

    vals, grads = prop.eval_and_grad(random_params)

Inspecting the symbolic expression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``.expression(idx)`` reconstructs the closed-form SymPy expression for
observable ``idx``.

.. code-block:: python

    prop.expression(0)  # ⟨Z₀⟩

.. code-block:: text

    -1.0*sin(θ₂)*sin(θ₃)*sin(θ₄)*cos(θ₀)*cos(θ₁) + 1.0*cos(θ₀)*cos(θ₂)*cos(θ₄)