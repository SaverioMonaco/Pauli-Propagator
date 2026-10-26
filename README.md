# Pauli Propagation of Parametrized Circuits
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23020043.svg)](https://doi.org/10.5281/zenodo.23020043)

Try it without installation via Binder: [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/SaverioMonaco/Pauli-Propagator/HEAD?urlpath=%2Fdoc%2Ftree%2Fnotebooks%2Fshowcase.ipynb)

## Overview

Convert the expectation values of a circuit’s observables into explicit functions of the circuit’s parameters.

## Usage

* First import all the necessary libraries:

```python
import pennylane as qml # To define the circuit
from pprop.propagator import Propagator # For Pauli Propagation
```

* Define the circuit as a function of the parameters

```python
# Function of parameters
def ansatz(params : list[float]):
    qml.RX(params[0], wires=0)
    qml.RX(params[1], wires=1)

    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=1)

    qml.Hadamard(wires = 2)

    qml.Barrier()

    qml.CNOT(wires = [0, 1])
    qml.CNOT(wires = [1, 2])

    qml.Barrier()

    qml.RY(params[4], wires=0)
    qml.RY(params[5], wires=1)
    qml.RY(params[6], wires=2)
    
    return [qml.expval(qml.PauliZ(qubit)) for qubit in range(1)] + [qml.expval(qml.PauliX(0)@qml.PauliX(1)@qml.PauliX(2))] + [qml.expval(qml.PauliY(2))] + [qml.expval(-qml.PauliX(0)@qml.PauliX(1)@qml.PauliX(2) + 13*qml.PauliZ(2))]
```

* Define the propagator
```python
prop = Propagator(
    ansatz, 
    k1 = None, # Cutoff on the Pauli Weight
    k2 = None, # Cutoff on the frequencies
    k2_mode = "frequency", # or "sine": k2 counts only sin factors (exact at theta = 0)
)
```
``` 
>>> prop
Propagator
  Number of qubits : 3
  Trainable parameters : 7
```
``` 
>>> prop.show()
0: ──RX(0.00)──RY(2.00)──||─╭●─────||──RY(4.00)─┤  <Z> ╭<X@X@X>      ╭<𝓗(-1.00,13.00)>
1: ──RX(1.00)──RY(3.00)──||─╰X─╭●──||──RY(5.00)─┤      ├<X@X@X>      ├<𝓗(-1.00,13.00)>
2: ──H───────────────────||────╰X──||──RY(6.00)─┤      ╰<X@X@X>  <Y> ╰<𝓗(-1.00,13.00)>
```

* Propagate the observables (this runs in the Rust extension `pprop_rs`):
```python
prop.propagate()
```

* Get the output expectation values by calling the class and feeding the circuit's parameters
```python
random_params = qml.numpy.arange(prop.num_params)
prop_output = prop(random_params)
[ 0.32448205 -0.52806187  0.          4.1604633 ]
```

* You can inspect the explicit functions using `.expression()`
```
>>> prop.expression(0)
-1.0*sin(θ2)*sin(θ3)*sin(θ4)*cos(θ0)*cos(θ1) + 1.0*cos(θ0)*cos(θ2)*cos(θ4)
```

## Installation

```bash
pip install pauli-propagator
```

This pulls in the Rust extension `pprop_rs` as a prebuilt wheel (Linux x86_64/aarch64,
macOS arm64, Windows x64; Python 3.12+). On other platforms pip builds it from
source, which needs a Rust toolchain ([rustup](https://rustup.rs/)).

### Running on IBM hardware (optional)

`pprop.qasm` exports a trained circuit to OpenQASM 3 with no extra
dependencies. To load it into Qiskit (`pprop.qasm.to_qiskit`) and run it on
IBM Quantum devices, install the `qiskit` extra, which adds `qiskit`,
`qiskit-qasm3-import` and `qiskit-ibm-runtime`:

```bash
pip install "pauli-propagator[qiskit]"
```

For a development install, use `pip install -e ".[qiskit]"` in the last step below.

### Development install

```bash
pip install maturin
cd native/pprop_rs
maturin develop --release
cd ../..
pip install -e .
```

`maturin develop` builds `pprop_rs` and installs it straight into the active
virtual environment. Re-run it after pulling changes to `native/pprop_rs`.

## Examples

For detailed examples, please refer to the `notebooks` directory, which
contains Jupyter notebooks that demonstrate various use cases and tests of
the propagator.

## Citation
If you use this software in your research or publications, **please cite** the following: 

```
@software{monaco_2026_23020043,
    author    = {Monaco, Saverio and
                 Lipponen, Aapo},
    doi       = {10.5281/zenodo.23021244},
    month     = sep,
    publisher = {Zenodo},
    swhid     = {swh:1:dir:ce152d847302ce6b14f1197f106986050d09c0f4
                 ;origin=https://doi.org/10.5281/zenodo.16028009;vi
                 sit=swh:1:snp:a9b8558798b8b9e8375e83e7aca935976731
                 18b2;anchor=swh:1:rel:6f6cac02e75daff2e374b752bb49
                 f157960d3915;path=SaverioMonaco-Pauli-
                 Propagator-4bb7e0d
                 },
    title     = {Pauli-Propagator},
    url       = {https://doi.org/10.5281/zenodo.23021244},
    version   = {v3.1.0},
    year      = 2026
}
```
