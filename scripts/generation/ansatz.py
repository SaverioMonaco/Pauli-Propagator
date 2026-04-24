import pennylane as qml


def ansatz(params, side):
    index = 0

    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1

    # Horizontal entanglers (CZ)
    for d in range(2):
        y_start = 0 if d % 2 == 0 else 1
        for x in range(side):
            for y in range(y_start, side - 1, 2):
                i = x * side + y
                j = x * side + (y + 1)
                qml.CZ(wires=[i, j])

    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1

    # Vertical entanglers (CZ)
    for d in range(2):
        x_start = 0 if d % 2 == 0 else 1
        for y in range(side):
            for x in range(x_start, side - 1, 2):
                i = x * side + y
                j = (x + 1) * side + y
                qml.CZ(wires=[i, j])

    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1
        qml.RY(params[index], wires=q)
        index += 1
