import pennylane as qml


def hamiltonian(side: int, J: float, h: float) -> qml.Hamiltonian:
    coeffs = []
    obs = []
    
    num_qubits = side * side

    # Nearest-neighbor ZZ interactions
    for x in range(side):
        for y in range(side):
            i = x * side + y

            # Right neighbor
            if y < side - 1:
                j = x * side + (y + 1)
                coeffs.append(-J / num_qubits)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

            # Down neighbor
            if x < side - 1:
                j = (x + 1) * side + y
                coeffs.append(-J / num_qubits)
                obs.append(qml.PauliZ(i) @ qml.PauliZ(j))

    # Transverse-field X terms
    for i in range(num_qubits):
        coeffs.append(-h / num_qubits)
        obs.append(qml.PauliX(i))

    return qml.Hamiltonian(coeffs, obs)