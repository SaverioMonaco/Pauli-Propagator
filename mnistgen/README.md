# mnistgen

Train a quantum circuit to generate binarised MNIST digits with Pauli
propagation, then sample it on IBM hardware.

```
train.py ──► result.json + circuit.qasm ──► deploy.py ──► samples.json
```

Run every script from inside `mnistgen/`. Shared code lives in the package:
`pprop.mnist` (data) and `pprop.qasm` (circuit export).

---

## `model.py`

Not a script; the other scripts import it. It defines the circuit: a
shell-brick ansatz on a `side x side` qubit grid, where qubit `r*side + c` is
pixel `(r, c)`:

```
RX → horizontal CZ brickwork → RX → vertical CZ brickwork → RX RY
```

It has `4 * side²` parameters. It also provides `z_string(qubits)` and
`circuit(side, observables)`, which is the callable `pprop.Propagator` needs.

---

## `train.py`

Trains the circuit by minimising a stochastic MMD loss. Each round it draws
`--num_obs` random PauliZ strings, propagates them with pprop (truncated at
`--k1`/`--k2`), and runs `--num_steps` Adam steps. The loss matches the
circuit's `<Z…Z>` to the same moments of the training images. At the end it
reports a batch MMD on the test split.

```bash
python train.py --side 8 --max_train 5000 --num_obs 1000 --num_rounds 100 --num_steps 1000 --lr 5e-5 --k1 12 --seed 0
```

| Arg | Default | Meaning |
|-----|---------|---------|
| `--side` | 8 | grid side, `num_qubits = side²` |
| `--digits` | 0–9 | digit classes to learn |
| `--max_train` | 5000 | number of training images |
| `--num_obs` | 20 | Z strings per round |
| `--num_rounds` | 100 | rounds; each draws a fresh batch of Z strings |
| `--num_steps` | 1000 | Adam steps per round |
| `--lr` | 0.1 | Adam learning rate |
| `--k1` | 6 | Pauli-weight truncation |
| `--k2` | none | trigonometric-degree truncation (off unless given) |
| `--seed` | 0 | seed for init and observable sampling |
| `--path` | `results` | output root |
| `--wandb` | off | W&B project name to log to |

**Writes** `results/side8_d0123456789_obs1000_lr5e-05_k12/000/`:
- `result.json`: params, losses, test MMD, and the full config
- `circuit.qasm`: the trained circuit, ready for hardware

For a quick check that everything works (a few seconds):
`python train.py --side 3 --digits 0 1 --num_rounds 2 --num_steps 20`

---

## `deploy.py`

Loads `circuit.qasm` from a training run, transpiles it for an IBM backend,
and samples bitstrings with SamplerV2. It needs `qiskit`,
`qiskit-ibm-runtime` and saved IBM credentials.

```bash
python deploy.py results/side8_d0123456789_obs1000_lr5e-05_k12/000 --backend ibm_berlin --shots 64 --optimization_level 1
```

| Arg | Default | Meaning |
|-----|---------|---------|
| `run` | required | training run directory |
| `--backend` | `ibm_torino` | IBM backend |
| `--channel` | `ibm_cloud` | IBM Runtime channel |
| `--shots` | 1024 | number of samples |
| `--optimization_level` | 1 | transpiler optimisation level |

**Writes** `<run>/samples.json`, which holds the backend, job id, bitstrings
and counts. Bitstrings are in Qiskit order, with qubit 0 as the rightmost
character.

---

## Conventions

- A pixel `x ∈ {0,1}` corresponds to the Z eigenvalue `1 − 2x`.
- MNIST is downloaded to `~/.cache/pprop/mnist`. Set `PPROP_MNIST_DIR` to
  change this.
- The parameter layout matches `scripts/generation/ansatz.py`, so old
  `result.json` params still work with `model.ansatz`.
