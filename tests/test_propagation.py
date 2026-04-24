# %%

from pprop import Propagator #noqa
import pennylane as qml
import random
import numpy as np

num_qubits = 3

# Single qubit no param gates
sqnp_gates = [qml.H, qml.S, qml.T, qml.SX]

# Single qubit param gates
sqp_gates = [qml.RX, qml.RY, qml.RZ]

# Two qubits no param gates
tqnp_gates = [qml.CNOT, qml.CY, qml.CZ]
                
# %%
def get_random_ansatz():
    # Build the circuit structure once, at definition time
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
            qml.expval(13*qml.PauliZ(2) + qml.PauliZ(0) @ qml.PauliX(1))
        ]

    return ansatz

# %%
def test_propagation():
    device = qml.device("default.qubit", wires=num_qubits)
    for a in range(3):
        ansatz = get_random_ansatz()
        qnode = qml.QNode(ansatz, device)

        propagator = Propagator(ansatz)
        propagator.propagate()

        for _ in range(5):
            random_params = qml.numpy.random.uniform(-np.pi, np.pi, propagator.num_params)
            prop_output = propagator(random_params)
            qml_output = qnode(random_params)

            assert np.allclose(prop_output, qml_output, atol=1e-6), (
                f"Mismatch:\nprop:  {prop_output}\nqml:   {qml_output}"
            )
# %%
test_propagation()