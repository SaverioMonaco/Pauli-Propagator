import pennylane as qml


def ansatz(params, side):
    index = 0

    # Initial layer: SX + RZ (native, no wasted leading RZ on |0>)
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

    # RX layer
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

    # Final RZ + SX + RZ layer
    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1
        qml.RY(params[index], wires=q)
        index += 1

if __name__ == '__main__':
    import numpy as np
    def exact_gradient_variance(prop, param_idx):
        """
        Exact gradient variance for parameter param_idx via the theorem:
            Var(dL/d theta_j) = sum_{x: j in S_x} c_x^2 * 2^{-|S_x|}
        Computed directly from prop.exprs[0] in O(|exprs|). No sampling.
        """
        var = 0.0
        for coeff, sin_idx, cos_idx in prop.exprs[0]:
            S_x = list(sin_idx) + list(cos_idx)
            if param_idx in S_x:
                var += coeff**2 * 2**(-len(S_x))
        return var

    def exact_gradient_variance_mean(prop):
        """
        Exact variance averaged over all parameter indices.
        """
        var_per_param = np.array([
            exact_gradient_variance(prop, j)
            for j in range(prop.num_params)
        ])
        return float(np.mean(var_per_param)), var_per_param

    import time #noqa

    from pprop import Propagator
    from pprop.propagator.pruning import DeadQubitPruner, XYWeightPruner #noqa
    
    side = 8 
    wires = np.random.choice(np.arange(side*side), 2, replace=False)
    
    def circuit(params):
        ansatz(params, side)
        return qml.expval(qml.prod(*[qml.PauliZ(wires=w) for w in wires]))

    prop = Propagator(circuit, k1=10, k2=40)
    print(prop)

    start = time.time()
    prop.propagate(pruners=[XYWeightPruner(), DeadQubitPruner()])
    end = time.time()

    print(f"  Random obs. {qml.prod(*[qml.PauliZ(wires=w) for w in wires])}")
    print(f"  Propagation time: {end - start:.3f} seconds")
    print(f"  Gradient variance: {float(exact_gradient_variance_mean(prop)[0])}")