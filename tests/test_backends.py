# %%
"""
Verifies that Propagator's three evaluator backends ("standard", "sparse",
"vmap") produce identical values and gradients against PennyLane's own
param-shift gradient, and against each other. They are alternative internal
representations of the exact same math (see propagate()'s docstring and
notebooks/test/sparse_arrays_explained.ipynb) - this test is what "identical"
is supposed to mean in practice.
"""
import random

import numpy as np
import pennylane as qml

from pprop import Propagator  # noqa

num_qubits = 3

sqnp_gates = [qml.H, qml.S, qml.T, qml.SX]
sqp_gates = [qml.RX, qml.RY, qml.RZ]
tqnp_gates = [qml.CNOT, qml.CY, qml.CZ]


# %%
def get_random_ansatz():
    layers = []
    for _ in range(5):
        single_gates = []
        for qubit in range(num_qubits):
            gate = random.choice(sqnp_gates + sqp_gates)
            single_gates.append((gate, qubit))
        gate = random.choice(tqnp_gates)
        q0, q1 = random.sample(range(num_qubits), 2)
        layers.append((single_gates, (gate, q0, q1)))

    def ansatz(params):
        param_idx = 0
        for single_gates, (tq_gate, q0, q1) in layers:
            for gate, qubit in single_gates:
                if gate in sqp_gates:
                    gate(params[param_idx], wires=qubit)
                    param_idx += 1
                else:
                    gate(wires=qubit)
            tq_gate(wires=[q0, q1])

        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2)),
            qml.expval(13 * qml.PauliZ(2) + qml.PauliZ(0) @ qml.PauliX(1)),
        ]

    return ansatz


# %%
def test_backends_agree_with_qml_and_each_other():
    device = qml.device("default.qubit", wires=num_qubits)

    for _ in range(3):
        ansatz = get_random_ansatz()
        qnode = qml.QNode(ansatz, device)

        propagators = {}
        for backend in ("standard", "sparse", "vmap"):
            propagators[backend] = Propagator(ansatz)
            propagators[backend].propagate(backend=backend)

        for _ in range(5):
            random_params = qml.numpy.random.uniform(-np.pi, np.pi, propagators["standard"].num_params)
            qml_output = qnode(random_params)
            qml_grad = qml.gradients.param_shift(qnode)(random_params)

            results = {
                backend: prop.eval_and_grad(random_params)
                for backend, prop in propagators.items()
            }

            for backend, (prop_output, prop_grad) in results.items():
                assert np.allclose(prop_output, qml_output, atol=1e-6), (
                    f"[{backend}] Mismatch EVAL vs qml:\nprop: {prop_output}\nqml:  {qml_output}"
                )
                assert np.allclose(prop_grad, qml_grad, atol=1e-6), (
                    f"[{backend}] Mismatch GRAD vs qml:\nprop: {prop_grad}\nqml:  {qml_grad}"
                )

            # And explicitly cross-check the backends against each other, not
            # just transitively via qml - this is the guarantee propagate()'s
            # docstring makes ("all three compute exactly the same values").
            std_vals, std_grads = results["standard"]
            for backend in ("sparse", "vmap"):
                vals, grads = results[backend]
                assert np.allclose(vals, std_vals, atol=1e-8), (
                    f"[{backend}] value mismatch vs standard: {vals} vs {std_vals}"
                )
                assert np.allclose(grads, std_grads, atol=1e-6), (
                    f"[{backend}] gradient mismatch vs standard"
                )


def test_invalid_backend_raises():
    ansatz = get_random_ansatz()
    propagator = Propagator(ansatz)
    try:
        propagator.propagate(backend="not-a-real-backend")
        raise AssertionError("expected ValueError for an invalid backend")
    except ValueError:
        pass


def test_eval_n_jobs_matches_single_threaded():
    ansatz = get_random_ansatz()
    device = qml.device("default.qubit", wires=num_qubits)
    qnode = qml.QNode(ansatz, device)

    for backend in ("standard", "sparse"):
        prop_serial = Propagator(ansatz)
        prop_serial.propagate(backend=backend, eval_n_jobs=1)

        prop_threaded = Propagator(ansatz)
        prop_threaded.propagate(backend=backend, eval_n_jobs=4)

        random_params = qml.numpy.random.uniform(-np.pi, np.pi, prop_serial.num_params)
        vals_serial, grads_serial = prop_serial.eval_and_grad(random_params)
        vals_threaded, grads_threaded = prop_threaded.eval_and_grad(random_params)

        assert np.allclose(vals_serial, vals_threaded, atol=1e-12)
        assert np.allclose(grads_serial, grads_threaded, atol=1e-12)


# %%
test_backends_agree_with_qml_and_each_other()
test_invalid_backend_raises()
test_eval_n_jobs_matches_single_threaded()
