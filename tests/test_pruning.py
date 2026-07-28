# %%
import time

import pennylane as qml
from numpy import mean, std

from pprop import Propagator

num_qubits = 6
num_iterations = 20

def ansatz(params):
    for i in range(num_qubits//2):
        qml.RX(params[i], wires=i)
    
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i+1])

    for i in range(num_qubits):
        qml.RX(params[i], wires=i)
        qml.RY(params[i], wires=i)

    return [qml.expval(qml.PauliZ(0) @ qml.PauliZ(num_qubits-1))]

time_opt = []
time_no_opt = []
# %%
for i in range(num_iterations):
    for use_dead, use_xy in [(False, False), (True, True), (False, True), (True, False)]:
        prop = Propagator(ansatz)
        time_start = time.time()
        prop.propagate(use_dead_qubit_pruner=use_dead, use_xy_weight_pruner=use_xy)
        time_stop = time.time()

        if use_dead or use_xy:
            time_opt.append(time_stop - time_start)
        else:
            time_no_opt.append(time_stop - time_start)


# %%
print( "           AVG       STD")
print( "________________________________")
print(f"Opt True : {mean(time_opt):.5f}   {std(time_opt):.5f}")
print(f"Opt False: {mean(time_no_opt):.5f}   {std(time_no_opt):.5f}")
print()
print(f"Opt Speedup: {100 * mean(time_no_opt) / mean(time_opt):.2f}%")