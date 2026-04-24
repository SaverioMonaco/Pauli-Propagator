# Scripts

Two experiment pipelines built on `pprop`: a **VQE** solver for the 2D transverse-field Ising model, and a **quantum generative model** trained on binarised MNIST.

Both pipelines follow the same two-step flow:

```
train.py  →  result.json
          ↘  deploy.py  →  samples/deploy.json
```

---

## VQE (`scripts/vqe/`)

Minimises the energy of the 2D transverse-field Ising Hamiltonian on an 8×8 lattice using a parameterised quantum circuit and analytic gradients via Pauli propagation.

### Files

| File | Role |
|------|------|
| `ansatz.py` | Shell-brick ansatz: RX layers interleaved with horizontal and vertical CZ brickwork. |
| `ising2d.py` | Builds the nearest-neighbour ZZ + transverse-X Hamiltonian as a `qml.Hamiltonian`. |
| `train.py` | Propagates the circuit and runs Adam to minimise the energy. |
| `deploy.py` | Transpiles the trained circuit for an IBM backend and evaluates the energy via `EstimatorV2`. |

### Quick start

**1. Train:**

```bash
cd scripts/vqe
python train.py --side 8 --J 1.0 --h 1.0 \
                --k1 64 --k2 256 \
                --lr 5e-3 --num_steps 1000 --seed 0
```

| Key argument | Default | Meaning |
|---|---|---|
| `--side` | `8` | Grid side (`num_qubits = side²`) |
| `--J` | `1.0` | ZZ coupling constant |
| `--h` | `1.0` | Transverse field strength |
| `--k1` | `64` | Pauli-weight cutoff |
| `--k2` | `256` | Frequency cutoff |
| `--lr` | `5e-3` | Adam learning rate |
| `--num_steps` | `1000` | Number of optimisation steps |
| `--seed` | `0` | RNG seed for parameter initialisation |
| `--path` | `.` | Root output directory |

Output directory: `<path>/side{side}_k{k1}_v{k2}/result.json`

`result.json` contains:

```json
{
  "params":          [...],
  "fun":             -1.85,
  "loss_history":    [...],
  "params_history":  [[...], ...],
  "num_qubits":      64,
  "J":               1.0,
  "h":               1.0
}
```

**2. Deploy to IBM Quantum** (requires IBM Cloud credentials):

```bash
python deploy.py --path ./side8_k64_v256 --backend ibm_berlin
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--path` | *(required)* | Run directory containing `result.json` |
| `--backend` | `ibm_berlin` | IBM backend name |
| `--channel` | `ibm_cloud` | IBM Runtime channel |

Submits Estimator jobs for 10 evenly-spaced checkpoints from the optimisation history and writes `deploy.json` alongside.

---

## Generation (`scripts/generation/`)

Trains a parameterised quantum circuit to match the distribution of binarised, downsampled MNIST digits by minimising a stochastic MMD (Maximum Mean Discrepancy) loss.

### Files

| File | Role |
|------|------|
| `ansatz.py` | Same shell-brick ansatz used by the VQE pipeline. |
| `train.py` | Stochastic MMD training loop: samples random Pauli-Z observables each round, propagates, and runs Adam. |
| `deploy.py` | Transpiles the trained circuit for an IBM backend and draws bitstring samples via `SamplerV2`. |

### Quick start

**1. Train:**

```bash
cd scripts/generation
python train.py --side 8 --num_obs 20 --num_epochs 100 \
                --lr 0.1 --num_steps 10000 \
                --k1 6 --k2 20 --seed 0
```

| Key argument | Default | Meaning |
|---|---|---|
| `--side` | `8` | Grid side (`num_qubits = side²`) |
| `--num_obs` | `20` | Pauli observables sampled per training round |
| `--num_epochs` | `100` | Number of rounds (each gets a fresh observable batch) |
| `--num_steps` | `10000` | Adam steps per round |
| `--lr` | `0.1` | Adam learning rate |
| `--k1` | `6` | Pauli-weight cutoff |
| `--k2` | `20` | Frequency cutoff |
| `--seed` | `6` | RNG seed |
| `--path` | `./results` | Root output directory |
| `--data_dir` | `mnist_data` | MNIST IDX files (downloaded automatically if absent) |

Output structure:

```
results/
└── 8_20_100_0.1_10000_6_20/   ← side_numobs_epochs_lr_steps_k1_k2
    └── 000/                    ← auto-incrementing run index
        └── result.json
```

`result.json` contains:

```json
{
  "num_qubits":   64,
  "params":       [...],
  "loss_history": [...],
  "mmd_mean":     0.012,
  "mmd_std":      0.003
}
```

**2. Deploy** (optional, requires IBM Cloud credentials):

```bash
cd scripts/generation
python deploy.py --path ./results/8_20_100_0.1_10000_6_20/000 \
                 --num_shots 64 --backend ibm_torino
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--path` | *(required)* | Run directory containing `result.json` |
| `--num_shots` | `64` | Measurement shots |
| `--backend` | `ibm_torino` | IBM backend name |
| `--channel` | `ibm_cloud` | IBM Runtime channel |

Writes `samples.json` (bitstrings + counts) into the run directory.

### Hyperparameter sweeps (SLURM)

`scripts/test/TRAIN.sh` and `TRAIN_mini.sh` show how to launch sweeps over learning rates and seeds via `sbatch`. Update the `LOG_DIR` and `--path` argument in `train.py` before submitting.

---

## Dependencies

Both pipelines require:

```
pprop          # this package (pip install -e .)
pennylane
numpy
```

Optional:

```
qiskit qiskit-ibm-runtime      # deploy (VQE and generation)
```
