# pprop_rs

The Rust core of pprop, built as the Python module `pprop_rs`. You do not call
it directly: `pprop.Propagator` does.

It has two engines:

- **tape** (default): the propagation is compiled once into a list of steps on
  numbered boxes, one box per Pauli word. `p(theta)` and
  `p.eval_and_grad(theta)` run those steps.
- **terms** (the original engine): every Pauli path is kept as its own term,
  a coefficient times a product of sines and cosines. Only used for
  `p.exprs`, the readable expression, which is built on request and is only
  practical for small circuits.

## Modules

```text
src/
├── lib.rs           the Python interface only: the functions and the class Python calls
├── pauli.rs         a Pauli word as two bit arrays (x plane, z plane), one bit per qubit
├── gates.rs         gate kind codes, GateSpec, sin/cos of fixed angles, and the rule
│                    tables (what each gate does to a word's letters)
├── pruning.rs       the exact pruners: dead qubit, XY weight
├── threads.rs       splitting the work over threads
├── pybuf.rs         reading NumPy arrays passed from Python
├── tape/            default engine: compile once, evaluate many times
│   ├── mod.rs       the data (Tape, Step, Block, TapeProgram) and the entry points
│   ├── compile.rs   the propagation, writing steps on boxes instead of computing numbers;
│   │                k1, pruners and coeff_threshold drop words here
│   ├── cleanup.rs   drops every step that never reaches the readout, renumbers the boxes
│   ├── eval.rs      the value: one pass over the steps
│   ├── grad.rs      value and gradient: the same pass, then one pass backwards
│   └── orders.rs    evaluation with the k2 truncation (k2 + 1 numbers per box)
└── terms/           original engine, only for p.exprs
    ├── mod.rs       the propagation keeping every path as a term
    ├── arena.rs     how a term's chain of sin/cos factors is stored
    ├── evolve.rs    what one gate does to one word and its terms
    └── evaluator.rs evaluating the sum of terms (value and gradient)
```

## What runs on each call

| Python | Rust entry point (`lib.rs`) | Files |
|---|---|---|
| `p.propagate()` | `compile_tapes` | `tape/compile.rs`, then `tape/cleanup.rs`; returns one `TapeProgram` per observable |
| `p(theta)` | `TapeProgram.eval` | `tape/eval.rs` (`tape/orders.rs` with k2) |
| `p.eval_and_grad(theta)` | `TapeProgram.eval_and_grad` | `tape/grad.rs` (`tape/orders.rs` with k2) |
| `p.exprs` | `propagate_batch` | `terms/*` |
| tests, debugging | `tape_debug` | the raw tape as a dict (boxes, words, steps, pairs, ...) |
| tests | `evolve_single_gate_debug` | one gate on one word (old engine), checked against the Python rule dicts |

## Building and testing

```bash
cd native/pprop_rs
cargo build --release                 # then copy target/release/libpprop_rs.so
                                      # to python/pprop_rs/pprop_rs.abi3.so
# or: maturin develop --release
cargo test                            # Rust unit tests
```

From the repository root, `pytest tests/` runs the Python tests. The ones
about this crate:

- `test_tape_compile.py`, `test_tape_eval.py`, `test_tape_grad.py`,
  `test_tape_k2.py`, `test_tape_threshold.py`, `test_tape_cleanup.py`: the tape
  engine, each against the old engine or by hand.
- `test_default_engine.py`: `Propagator` uses the tape and agrees with `exprs`.
- `test_rule_tables.py`: the Rust rule tables against the `rule` dicts in
  `src/pprop/gates/*.py`, entry by entry.
- `test_backends.py`: whole random circuits against PennyLane.

## Adding a gate

1. **Python.** Add the gate class in `src/pprop/gates/` with its `rule` dict
   (the readable reference), and its kind code in `_GATE_KIND` in
   `src/pprop/propagator/__init__.py`.
2. **`gates.rs`.** Add the same kind code as a constant, its rule table
   (`xx_rule`, transcribed from the Python `rule` dict), and an arm in
   `gate_acts`. A fixed angle goes through `fixed_sin_cos`.
3. **`tape/compile.rs`.** Tell the compiler which shape the gate has:
   - a Clifford gate (a word goes to one word, maybe with a sign): add it to
     `is_clifford` and `clifford_image`;
   - a rotation on one qubit (pairs two words with cos/sin): add it to
     `rotation_rule` (T is compiled this way, as RZ with a fixed angle);
   - a controlled rotation (groups of up to 4 words): add it to
     `is_controlled_rotation` and `cr_rule_of`.

   A gate with a new shape needs a new kind of `Step` in `tape/mod.rs`, and
   its handling in `eval.rs`, `grad.rs`, `orders.rs` and `cleanup.rs`.
4. **`terms/evolve.rs`.** Add an arm in `evolve_one_gate`, so that `p.exprs`
   works too.
5. **Tests.** Add the gate to `test_rule_tables.py`, to the random circuits of
   `test_backends.py`, and to `every_gate_case` in `test_tape_grad.py` (used
   by the tape tests).
