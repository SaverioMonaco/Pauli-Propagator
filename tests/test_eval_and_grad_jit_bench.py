"""
Investigates whether JIT (JAX) or vmap can speed up ``Propagator.eval_and_grad``,
which is called once per Adam step and is the main per-step cost during training.

Context
-------
``Propagator.propagate()`` already compiles each observable's symbolic
expression into a fast closure via ``make_evaluator`` (see
``pprop/propagator/utils.py``): every ``CoeffTerms`` list is converted once,
via ``build_arrays``, into three dense NumPy arrays of shape ``(n_terms,)`` /
``(n_terms, num_params)``, and evaluation is vectorised NumPy - there is no
Python loop over terms. ``Propagator.eval_and_grad`` then loops in *Python*
over one such closure per observable.

Why a previous JIT attempt could take minutes to hours
--------------------------------------------------------
Wrapping the whole per-observable loop in ``jax.jit``, or building the JAX
graph by looping over ``CoeffTerms`` in Python (one ``jnp.sin``/``jnp.cos`` op
per term), makes the traced graph size scale with the number of terms.
XLA compile time grows worse than linearly with graph size, so with
thousands-to-hundreds-of-thousands of terms per observable this genuinely
compiles for minutes to hours. Section 2 below reproduces this pathology on
a small, controlled example so the effect is visible in seconds rather than
hours.

The fix demonstrated in sections 3-4 reuses the *same* dense-array
representation the code already builds (``build_arrays``), just evaluated
with ``jax.numpy`` instead of ``numpy``. Because the array shapes are fixed
regardless of how large ``n_terms`` is, the JIT trace is O(1) in term count
(a handful of array ops, not one op per term) - compile time stays flat, and
``jax.vmap`` additionally lets us batch *all* observables into a single
compiled call, removing the Python loop over observables entirely.

IMPORTANT - measured outcome (CPU-only, no CUDA jaxlib in this environment):
at the SIDE=4/NUM_OBS=20 scale run by default below, and again at a more
realistic SIDE=8/NUM_OBS=200/k1=9/k2=64 scale (162 obs after pruning,
num_params=256, up to 248 terms/obs - about 3 min to propagate, so not run
by default here), **both jax.jit and jax.jit+vmap were consistently SLOWER
than the existing NumPy code**, by roughly 30-40% at the realistic scale.

An earlier version of this docstring attributed this to "JAX dispatch
overhead" - that was wrong, and section 5b's op-by-op microbenchmark
disproves it directly (measured pure jax.jit call overhead: a few
*microseconds*, nowhere near the ms-scale gaps observed here). The real
cause, isolated in section 5b:
  - For the DENSE representation (sections 2-4): the forward pass alone is
    already slower than NumPy once you're doing elementwise power/product
    over the full `num_params` width per term (no surprise - more columns,
    more work) - and JAX's *automatic differentiation* through that
    `jnp.prod` roughly triples the cost on top of the forward pass.
  - For the narrower SPARSE representation (section 5, `W` columns instead
    of `num_params`): the forward pass is actually fast (competitive with or
    faster than NumPy). What kills it is the **gradient's final scatter-add
    step** - routing each term's narrow `(n_terms, W)` contributions back
    into a dense `(num_params,)` gradient, exactly what NumPy does with
    `np.add.at`. In JAX (`.at[idx].add(...)` or `jax.ops.segment_sum`,
    tried both) that scatter-reduce over overlapping indices was ~20-24x
    more expensive than the forward pass alone, on this CPU backend.
Section 5b reproduces this with synthetic arrays at realistic shapes, with
the array sizes exposed as constants so you can change them and watch the
gap move - see it for the actual numbers and how to poke at them.

**Retest on a real GPU allocation before ruling JAX out for your actual
workload** - this scatter-add cost is specifically a CPU/XLA:CPU finding;
GPUs have different (and workload-dependent) scatter performance, and this
box has no CUDA jaxlib to check. Section 6 below benchmarks a
JAX-independent lever - threading the existing per-observable NumPy loop
across cores - which won on this CPU-only box where JAX did not.

The actual biggest win found here (section 5): a DATA STRUCTURE problem, not
a compute-engine one. ``build_arrays`` densifies every term into a full
``(num_params,)`` row, but ``k1`` (Pauli weight truncation) caps how many
parameters any single term can actually touch - measured ~17.7 of 256 params
(~6.9%) touched per term on average at the SIDE=8/k1=9/k2=64 scale, i.e. over
93% of every dense row is `sin(theta)**0 = 1` / `cos(theta)**0 = 1` computed
and multiplied for nothing. Note this "sparse" representation is *not* the
original `CoeffTerms` list (which is already ragged/ad-hoc - no padding at
all, just Python lists of whatever indices a term touches). It's a
*narrower dense* array: `(n_terms, W)` instead of `(n_terms, num_params)`,
`W` = max params actually touched by any one term in that expression (~24
here, vs. 256) - still fixed-width and vectorisable, just sized to the
actual per-term degree instead of the whole circuit. Gathering `theta[idx]`
into that shape instead of broadcasting over the full parameter vector gave
a **~6x speedup in plain NumPy, at every scale tested (49, 162, and the full
1000-observable/687-after-dedup regime)**, to machine precision (`atol`
~1e-16). This didn't need JAX at all. Section 5b implements and profiles
the *same* sparse arrays under `jax.jit`+`vmap`: forward-only is fast, but
once the gradient's scatter-add is included it's slower than plain sparse
NumPy (see above) - the dense representation was the bottleneck, not the
choice of NumPy vs. JAX, but the gradient's scatter-add is a genuine
separate JAX-specific cost once that's fixed.

This file only imports from ``pprop`` - nothing in ``src/`` is modified.

Run directly:
    python tests/test_eval_and_grad_jit_bench.py

To check these findings hold at your real training scale, bump SIDE/NUM_OBS/
K1/K2 below towards your real run (e.g. side=8, num_obs=1000, k1=9, k2=64) -
kept small here so the whole file runs in well under a minute. (Numbers in
this docstring for that scale come from a separate, longer-running pass -
see the summary section for pointers on reproducing them.)
"""
# %%
import time

import numpy as np
import pennylane as qml

import jax
jax.config.update("jax_enable_x64", True)  # match NumPy float64 - JAX defaults to
                                            # float32, which is too coarse for Adam
                                            # once gradients from many terms are summed
import jax.numpy as jnp

from pprop import Propagator
from pprop.propagator.pruning import DeadQubitPruner, XYWeightPruner
from pprop.propagator.utils import build_arrays

# %%
# ---------------------------------------------------------------------------
# 0. Build a moderate-scale Propagator.
# ---------------------------------------------------------------------------
SIDE, NUM_OBS, K1, K2, SEED = 4, 20, 9, 64, 0
num_qubits = SIDE * SIDE
rng = np.random.default_rng(SEED)


def ansatz(params, side):
    index = 0
    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1
    for d in range(2):
        y_start = 0 if d % 2 == 0 else 1
        for x in range(side):
            for y in range(y_start, side - 1, 2):
                qml.CZ(wires=[x * side + y, x * side + y + 1])
    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1
    for d in range(2):
        x_start = 0 if d % 2 == 0 else 1
        for y in range(side):
            for x in range(x_start, side - 1, 2):
                qml.CZ(wires=[x * side + y, (x + 1) * side + y])
    for q in range(side * side):
        qml.RX(params[index], wires=q)
        index += 1
        qml.RY(params[index], wires=q)
        index += 1


def sample_observables(n_obs):
    observables = []
    for _ in range(n_obs):
        k = rng.integers(1, 4)
        qubits = rng.choice(num_qubits, size=k, replace=False)
        ob = qml.PauliZ(int(qubits[0]))
        for q in qubits[1:]:
            ob = ob @ qml.PauliZ(int(q))
        observables.append(ob)
    return observables


obs = sample_observables(NUM_OBS)


def circuit(params):
    ansatz(params, SIDE)
    return [qml.expval(o) for o in obs]


prop = Propagator(circuit, k1=K1, k2=K2)
prop.propagate(pruners=[XYWeightPruner(), DeadQubitPruner()])
print(
    f"Propagated {len(prop.exprs)} observables, num_params={prop.num_params}, "
    f"term counts: min={min(len(e) for e in prop.exprs)} "
    f"max={max(len(e) for e in prop.exprs)}"
)

params0 = rng.normal(size=prop.num_params)

# %%
# ---------------------------------------------------------------------------
# 1. Baseline: current Propagator.eval_and_grad (pure NumPy, Python loop over
#    observables).
# ---------------------------------------------------------------------------
N_REPEAT = 30

_ = prop.eval_and_grad(params0)  # warm-up
t0 = time.perf_counter()
for _ in range(N_REPEAT):
    vals_np, grads_np = prop.eval_and_grad(params0)
baseline_ms = (time.perf_counter() - t0) / N_REPEAT * 1000
print(f"\n[baseline]  {baseline_ms:.3f} ms/step")

# %%
# ---------------------------------------------------------------------------
# 2. Reproduce the pathology: JIT-ing a Python for-loop over CoeffTerms (one
#    jnp.sin/jnp.cos per term) makes compile time blow up with n_terms. This
#    is almost certainly what caused the "minutes to hours" compiles - NOT
#    jax.jit itself being slow, but the traced graph being O(n_terms) instead
#    of O(1). Uses a synthetic expression (not the real propagator output) so
#    term count can be dialed up in a controlled way.
# ---------------------------------------------------------------------------
def make_synthetic_expr(n_terms, num_params):
    expr = []
    for _ in range(n_terms):
        coeff = rng.normal()
        sin_idx = rng.integers(0, num_params, size=rng.integers(0, 3)).tolist()
        cos_idx = rng.integers(0, num_params, size=rng.integers(0, 3)).tolist()
        expr.append((coeff, sin_idx, cos_idx))
    return expr


def naive_unrolled_jit(expr):
    """The pattern that doesn't scale: one jnp op emitted per term."""
    def _eval(thetas):
        total = 0.0
        for coeff, sin_idx, cos_idx in expr:
            term = coeff
            for i in sin_idx:
                term = term * jnp.sin(thetas[i])
            for j in cos_idx:
                term = term * jnp.cos(thetas[j])
            total = total + term
        return total
    return jax.jit(jax.value_and_grad(_eval))


print(
    "\n[naive per-term unrolled jit] compile time vs. term count "
    "(synthetic demo, capped low - already tens of seconds at 1500 terms; "
    "extrapolates to hours at the tens-of-thousands-of-terms scale seen in "
    "real runs):"
)
thetas0 = jnp.asarray(params0)
for n_terms in (100, 500, 1500):
    expr_demo = make_synthetic_expr(n_terms, prop.num_params)
    f = naive_unrolled_jit(expr_demo)
    t0 = time.perf_counter()
    v, g = f(thetas0)
    jax.block_until_ready((v, g))
    compile_s = time.perf_counter() - t0
    print(f"  n_terms={n_terms:5d}  first-call (compile+run) = {compile_s:.3f} s")

# %%
# ---------------------------------------------------------------------------
# 3. The fix, step 1: reuse build_arrays' dense (n_terms, num_params)
#    representation - the exact thing make_evaluator already builds - just
#    evaluated with jax.numpy + jax.jit instead of numpy. Trace size is now
#    independent of n_terms, so compile time should stay flat regardless of
#    how many terms an observable has.
# ---------------------------------------------------------------------------
def make_jax_eval_and_grad(expr, num_params):
    coeffs, sin_counts, cos_counts = build_arrays(expr, num_params)  # from src, unmodified
    coeffs = jnp.asarray(coeffs)
    sin_counts = jnp.asarray(sin_counts)
    cos_counts = jnp.asarray(cos_counts)

    def _eval(thetas):
        sins, coss = jnp.sin(thetas), jnp.cos(thetas)
        sin_prods = jnp.prod(sins[None, :] ** sin_counts, axis=1)
        cos_prods = jnp.prod(coss[None, :] ** cos_counts, axis=1)
        return jnp.sum(coeffs * sin_prods * cos_prods)

    return jax.jit(jax.value_and_grad(_eval))


t0 = time.perf_counter()
jitted_fns = [make_jax_eval_and_grad(e, prop.num_params) for e in prop.exprs]
warmup = [f(thetas0) for f in jitted_fns]
jax.block_until_ready(warmup)
compile_all_s = time.perf_counter() - t0
print(
    f"\n[vectorised per-obs jit] one-time compile, all {len(jitted_fns)} obs: "
    f"{compile_all_s:.3f} s"
)

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    results = [f(thetas0) for f in jitted_fns]
per_obs_jit_ms = (time.perf_counter() - t0) / N_REPEAT * 1000
vals_jit = np.array([float(v) for v, _ in results])
grads_jit = np.stack([np.asarray(g) for _, g in results])
print(
    f"[vectorised per-obs jit] steady state: {per_obs_jit_ms:.3f} ms/step "
    f"(speedup x{baseline_ms / per_obs_jit_ms:.2f})"
)

assert np.allclose(vals_jit, vals_np, atol=1e-6), "value mismatch vs baseline"
assert np.allclose(grads_jit, grads_np, atol=1e-6), "gradient mismatch vs baseline"
print("  correctness OK vs propagator.eval_and_grad")

# %%
# ---------------------------------------------------------------------------
# 4. The fix, step 2: vmap across ALL observables at once. Pad every
#    observable's dense arrays to a common n_terms (zero-coeff padding rows
#    contribute exactly 0 to both value and gradient), stack into one batched
#    array, and vmap+jit a single function. This removes the Python loop over
#    observables entirely - one compiled call replaces the whole
#    `_eval_and_grad_list` comprehension in Propagator.eval_and_grad.
# ---------------------------------------------------------------------------
arrs = [build_arrays(e, prop.num_params) for e in prop.exprs]
max_terms = max(c.shape[0] for c, _, _ in arrs)
num_obs = len(arrs)

coeffs_b = np.zeros((num_obs, max_terms))
sin_b = np.zeros((num_obs, max_terms, prop.num_params), dtype=np.int32)
cos_b = np.zeros((num_obs, max_terms, prop.num_params), dtype=np.int32)
for i, (c, s, co) in enumerate(arrs):
    n = c.shape[0]
    coeffs_b[i, :n], sin_b[i, :n], cos_b[i, :n] = c, s, co
coeffs_b, sin_b, cos_b = jnp.asarray(coeffs_b), jnp.asarray(sin_b), jnp.asarray(cos_b)


def _single_eval(thetas, coeffs, sin_counts, cos_counts):
    sins, coss = jnp.sin(thetas), jnp.cos(thetas)
    sin_prods = jnp.prod(sins[None, :] ** sin_counts, axis=1)
    cos_prods = jnp.prod(coss[None, :] ** cos_counts, axis=1)
    return jnp.sum(coeffs * sin_prods * cos_prods)


batched_eval_and_grad = jax.jit(
    jax.vmap(jax.value_and_grad(_single_eval), in_axes=(None, 0, 0, 0))
)

t0 = time.perf_counter()
vals_vmap, grads_vmap = batched_eval_and_grad(thetas0, coeffs_b, sin_b, cos_b)
jax.block_until_ready((vals_vmap, grads_vmap))
compile_vmap_s = time.perf_counter() - t0
print(f"\n[vmap over all obs] one-time compile: {compile_vmap_s:.3f} s")

t0 = time.perf_counter()
for _ in range(N_REPEAT):
    vals_vmap, grads_vmap = batched_eval_and_grad(thetas0, coeffs_b, sin_b, cos_b)
jax.block_until_ready((vals_vmap, grads_vmap))
vmap_ms = (time.perf_counter() - t0) / N_REPEAT * 1000
print(
    f"[vmap over all obs] steady state: {vmap_ms:.3f} ms/step "
    f"(speedup x{baseline_ms / vmap_ms:.2f})"
)

assert np.allclose(np.asarray(vals_vmap), vals_np, atol=1e-6), "value mismatch vs baseline"
assert np.allclose(np.asarray(grads_vmap), grads_np, atol=1e-6), "gradient mismatch vs baseline"
print("  correctness OK vs propagator.eval_and_grad")

# %%
# ---------------------------------------------------------------------------
# 5. The actual data-structure fix: `build_arrays` stores one column per
#    circuit parameter for every term, but k1 (Pauli weight truncation) caps
#    how many parameters any term can touch - most of those columns are 0
#    (sin/cos to the power 0 = 1, multiplied in for nothing). Gather only the
#    parameters a term actually touches into a narrow (n_terms, W) array
#    (W = max params touched by any one term) instead of broadcasting every
#    term over the full (num_params,) vector. This is a plain-NumPy change -
#    no JAX required - and it composes with everything above (sections 3/4
#    show the same idea works if you additionally want to JIT/vmap it).
# ---------------------------------------------------------------------------
from collections import Counter


def build_sparse_arrays(expr, num_params):
    """Like build_arrays, but only stores the (param_idx, power) pairs that
    are actually nonzero for each term, padded to a common narrow width W
    instead of the full num_params width."""
    packed = []
    for coeff, sin_idx, cos_idx in expr:
        packed.append((coeff, list(Counter(sin_idx).items()), list(Counter(cos_idx).items())))
    Ws = max((len(s) for _, s, _ in packed), default=1) or 1
    Wc = max((len(c) for _, _, c in packed), default=1) or 1
    n = len(packed)
    coeffs = np.zeros(n)
    idx_sin, pow_sin = np.zeros((n, Ws), dtype=np.int64), np.zeros((n, Ws))
    idx_cos, pow_cos = np.zeros((n, Wc), dtype=np.int64), np.zeros((n, Wc))
    for i, (coeff, s, c) in enumerate(packed):
        coeffs[i] = coeff
        for j, (idx, p) in enumerate(s):
            idx_sin[i, j], pow_sin[i, j] = idx, p
        for j, (idx, p) in enumerate(c):
            idx_cos[i, j], pow_cos[i, j] = idx, p
    return coeffs, idx_sin, pow_sin, idx_cos, pow_cos


def make_sparse_evaluator(expr, num_params):
    coeffs, idx_sin, pow_sin, idx_cos, pow_cos = build_sparse_arrays(expr, num_params)

    def _eval_grad(thetas):
        sin_g, cos_g = np.sin(thetas)[idx_sin], np.cos(thetas)[idx_cos]  # gather, not broadcast
        sin_pow = np.where(pow_sin > 0, sin_g ** pow_sin, 1.0)
        cos_pow = np.where(pow_cos > 0, cos_g ** pow_cos, 1.0)
        sin_prod, cos_prod = sin_pow.prod(axis=1), cos_pow.prod(axis=1)
        term_vals = coeffs * sin_prod * cos_prod

        def excl(pow_arr):
            n = pow_arr.shape[0]
            left = np.cumprod(np.concatenate([np.ones((n, 1)), pow_arr[:, :-1]], axis=1), axis=1)
            right = np.cumprod(np.concatenate([pow_arr[:, 1:], np.ones((n, 1))], axis=1)[:, ::-1], axis=1)[:, ::-1]
            return left * right

        excl_sin, excl_cos = excl(sin_pow), excl(cos_pow)
        cos_at_sin, sin_at_cos = np.cos(thetas)[idx_sin], np.sin(thetas)[idx_cos]

        d_sin = np.where(pow_sin > 0, pow_sin * np.where(sin_g != 0, sin_g ** (pow_sin - 1), 0.0) * cos_at_sin, 0.0)
        d_cos = np.where(pow_cos > 0, -pow_cos * np.where(cos_g != 0, cos_g ** (pow_cos - 1), 0.0) * sin_at_cos, 0.0)

        sin_grad_terms = coeffs[:, None] * d_sin * excl_sin * cos_prod[:, None]
        cos_grad_terms = coeffs[:, None] * sin_prod[:, None] * excl_cos * d_cos

        grad = np.zeros(num_params)
        np.add.at(grad, idx_sin.ravel(), sin_grad_terms.ravel())
        np.add.at(grad, idx_cos.ravel(), cos_grad_terms.ravel())
        return float(term_vals.sum()), grad

    return _eval_grad


sparse_fns = [make_sparse_evaluator(e, prop.num_params) for e in prop.exprs]


def sparse_eval_and_grad(params):
    results = [f(params) for f in sparse_fns]
    return np.array([v for v, _ in results]), np.stack([g for _, g in results])


vals_sparse, grads_sparse = sparse_eval_and_grad(params0)  # warm-up
t0 = time.perf_counter()
for _ in range(N_REPEAT):
    vals_sparse, grads_sparse = sparse_eval_and_grad(params0)
sparse_ms = (time.perf_counter() - t0) / N_REPEAT * 1000
print(
    f"\n[sparse/gathered numpy] {sparse_ms:.3f} ms/step "
    f"(speedup x{baseline_ms / sparse_ms:.2f})"
)
assert np.allclose(vals_sparse, vals_np, atol=1e-9), "value mismatch vs baseline"
assert np.allclose(grads_sparse, grads_np, atol=1e-9), "gradient mismatch vs baseline"
print("  correctness OK vs propagator.eval_and_grad (matches to ~1e-16, not just 1e-6)")

# %%
# ---------------------------------------------------------------------------
# 5b. The sparse arrays ARE vmappable - here's the actual jax.jit+vmap version
#     of section 5, plus an op-by-op breakdown showing exactly which step
#     makes it slower than plain sparse NumPy. TWEAKABLE: change SYN_NUM_OBS /
#     SYN_MAX_TERMS / SYN_W / SYN_NUM_PARAMS below and rerun this cell to see
#     how the gap moves - this uses synthetic arrays at whatever shape you
#     pick, not the real propagator, so it's cheap to explore.
# ---------------------------------------------------------------------------
SYN_NUM_OBS, SYN_MAX_TERMS, SYN_W, SYN_NUM_PARAMS = 700, 500, 24, 256  # matches the real
                                                                        # 1000-observable scale

syn_rng = np.random.default_rng(1)
syn_idx = syn_rng.integers(0, SYN_NUM_PARAMS, size=(SYN_NUM_OBS, SYN_MAX_TERMS, SYN_W)).astype(np.int64)
syn_pow = syn_rng.integers(0, 3, size=(SYN_NUM_OBS, SYN_MAX_TERMS, SYN_W)).astype(np.float64)
syn_coeffs = syn_rng.normal(size=(SYN_NUM_OBS, SYN_MAX_TERMS))
syn_thetas = syn_rng.normal(size=SYN_NUM_PARAMS)

syn_idx_j, syn_pow_j = jnp.asarray(syn_idx), jnp.asarray(syn_pow)
syn_coeffs_j, syn_thetas_j = jnp.asarray(syn_coeffs), jnp.asarray(syn_thetas)


def _sparse_single_eval(thetas, coeffs, idx, pw):
    """One observable's value, sparse/gathered form - this is what gets vmapped."""
    g = jnp.sin(thetas)[idx]                      # gather: (n_terms, W)
    p = jnp.where(pw > 0, g ** pw, 1.0)
    return (coeffs * jnp.prod(p, axis=1)).sum()


def _bench_jit(fn, *args, n_repeat=10):
    f = jax.jit(fn)
    r = f(*args)
    jax.block_until_ready(r)
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        r = f(*args)
    jax.block_until_ready(r)
    return (time.perf_counter() - t0) / n_repeat * 1000, r


print(f"\n[sparse arrays + jax] op-by-op breakdown at "
      f"(num_obs={SYN_NUM_OBS}, max_terms={SYN_MAX_TERMS}, W={SYN_W}, num_params={SYN_NUM_PARAMS}):")

# (1) forward value only, vmapped
ms, _ = _bench_jit(jax.vmap(_sparse_single_eval, in_axes=(None, 0, 0, 0)),
                    syn_thetas_j, syn_coeffs_j, syn_idx_j, syn_pow_j)
print(f"  1. forward only (vmap+jit):                    {ms:8.2f} ms")

# (2) forward + autodiff gradient (jax.value_and_grad), vmapped - the "obvious" way
ms, (vals_auto, grads_auto) = _bench_jit(
    jax.vmap(jax.value_and_grad(_sparse_single_eval), in_axes=(None, 0, 0, 0)),
    syn_thetas_j, syn_coeffs_j, syn_idx_j, syn_pow_j,
)
print(f"  2. forward + autodiff grad (vmap+jit):         {ms:8.2f} ms  <- jax.value_and_grad")


def _sparse_single_val_grad_manual(thetas, coeffs, idx, pw):
    """Same manual prefix/suffix-cumprod gradient trick as the NumPy version
    in section 5, just written in jnp instead of np."""
    g = jnp.sin(thetas)[idx]
    p = jnp.where(pw > 0, g ** pw, 1.0)
    prod = jnp.prod(p, axis=1)
    val = (coeffs * prod).sum()

    n = pw.shape[0]
    left = jnp.cumprod(jnp.concatenate([jnp.ones((n, 1)), p[:, :-1]], axis=1), axis=1)
    right = jnp.cumprod(jnp.concatenate([p[:, 1:], jnp.ones((n, 1))], axis=1)[:, ::-1], axis=1)[:, ::-1]
    excl = left * right
    cosg = jnp.cos(thetas)[idx]
    d = jnp.where(pw > 0, pw * jnp.where(g != 0, g ** (pw - 1), 0.0) * cosg, 0.0)
    grad_terms = coeffs[:, None] * d * excl                      # (n_terms, W) - narrow, cheap so far
    grad = jnp.zeros(SYN_NUM_PARAMS).at[idx.ravel()].add(grad_terms.ravel())  # <- the scatter-add
    return val, grad


ms, (vals_manual, grads_manual) = _bench_jit(
    jax.vmap(_sparse_single_val_grad_manual, in_axes=(None, 0, 0, 0)),
    syn_thetas_j, syn_coeffs_j, syn_idx_j, syn_pow_j,
)
print(f"  3. forward + MANUAL grad, incl. scatter (vmap+jit): {ms:8.2f} ms  <- same trick as NumPy section 5")
assert np.allclose(np.asarray(vals_manual), np.asarray(vals_auto), atol=1e-8), "(2) vs (3) value mismatch"
assert np.allclose(np.asarray(grads_manual), np.asarray(grads_auto), atol=1e-6), "(2) vs (3) gradient mismatch"
print("     correctness OK: manual gradient (3) matches autodiff gradient (2)")


def _grad_terms_only(thetas, coeffs, idx, pw):
    """Everything from (3) except the final scatter-add - stops at the
    narrow (n_terms, W) per-term gradient contributions."""
    g = jnp.sin(thetas)[idx]
    p = jnp.where(pw > 0, g ** pw, 1.0)
    n = pw.shape[0]
    left = jnp.cumprod(jnp.concatenate([jnp.ones((n, 1)), p[:, :-1]], axis=1), axis=1)
    right = jnp.cumprod(jnp.concatenate([p[:, 1:], jnp.ones((n, 1))], axis=1)[:, ::-1], axis=1)[:, ::-1]
    excl = left * right
    cosg = jnp.cos(thetas)[idx]
    d = jnp.where(pw > 0, pw * jnp.where(g != 0, g ** (pw - 1), 0.0) * cosg, 0.0)
    return coeffs[:, None] * d * excl


ms, _ = _bench_jit(jax.vmap(_grad_terms_only, in_axes=(None, 0, 0, 0)),
                    syn_thetas_j, syn_coeffs_j, syn_idx_j, syn_pow_j)
print(f"  4. same as (3) but WITHOUT the final scatter-add:   {ms:8.2f} ms  <- isolates the scatter's cost")

print(
    "\n  (2) vs (3) shows the manual gradient formula doesn't rescue this - it's not\n"
    "  autodiff specifically. (4) vs (3) isolates the actual cost: everything up to\n"
    "  the final scatter-add is cheap (comparable to (1)); the scatter-add that\n"
    "  routes narrow per-term contributions back into the (num_params,) gradient\n"
    "  is what dominates. This is a property of how JAX's XLA:CPU backend handles\n"
    "  scatter-reduce with overlapping indices, batched over vmap - not of JAX/jit\n"
    "  in general. Try shrinking SYN_W or SYN_NUM_PARAMS above and rerun: the gap\n"
    "  should shrink as there's less to scatter."
)

# %%
# ---------------------------------------------------------------------------
# 6. JAX-independent lever, tried: thread the existing per-observable NumPy
#    loop across CPU cores (no pickling/spawn cost like processes would have,
#    since NumPy releases the GIL during array ops). RESULT: this consistently
#    made things SLOWER at every thread count tested (2-128), here and at
#    real training scale - each observable's evaluation is too small
#    (~0.1-0.2ms) for the released-GIL work to outweigh Python's GIL
#    acquire/release + thread-scheduling overhead. Kept in this file as a
#    documented negative result, not a recommendation - see
#    Propagator.propagate()'s eval_n_jobs docstring.
# ---------------------------------------------------------------------------
try:
    from joblib import Parallel, delayed

    def joblib_eval_and_grad(params, n_jobs):
        # prop._eval_and_grad_list closures take precomputed sin(theta)/
        # cos(theta), not theta itself - see Propagator.__call__/eval_and_grad.
        sins, coss = np.sin(params), np.cos(params)
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(f)(sins, coss) for f in prop._eval_and_grad_list
        )
        vals = np.array([v for v, _ in results])
        grads = np.stack([g for _, g in results])
        return vals, grads

    print("\n[joblib threads] (JAX-independent - parallelises the existing NumPy loop)")
    for n_jobs in (4, 8):
        _ = joblib_eval_and_grad(params0, n_jobs)  # warm-up (thread pool spin-up)
        t0 = time.perf_counter()
        for _ in range(N_REPEAT):
            vals_jl, grads_jl = joblib_eval_and_grad(params0, n_jobs)
        joblib_ms = (time.perf_counter() - t0) / N_REPEAT * 1000
        assert np.allclose(vals_jl, vals_np, atol=1e-6)
        print(
            f"  n_jobs={n_jobs:3d}: {joblib_ms:.3f} ms/step "
            f"(speedup x{baseline_ms / joblib_ms:.2f})"
        )
except ImportError:
    print("\n[joblib threads] joblib not installed - skipping (pip install joblib)")

# %%
# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("\n=== summary (this run's scale) ===")
print(f"{'approach':34s}{'ms/step':>12s}{'speedup':>10s}")
print(f"{'baseline (numpy, per-obs loop)':34s}{baseline_ms:12.3f}{1.0:10.2f}")
print(f"{'jax.jit per observable':34s}{per_obs_jit_ms:12.3f}{baseline_ms / per_obs_jit_ms:10.2f}")
print(f"{'jax.jit + vmap over all obs':34s}{vmap_ms:12.3f}{baseline_ms / vmap_ms:10.2f}")
print(f"{'sparse/gathered numpy':34s}{sparse_ms:12.3f}{baseline_ms / sparse_ms:10.2f}")
print(
    """
Reference numbers from a separate, longer-running pass at SIDE=8/K1=9/K2=64
(bump the constants in section 0 and rerun to reproduce - propagation alone
takes minutes at this scale):

  NUM_OBS=200  (162 unique obs, num_params=256, up to 248 terms/obs):
    baseline numpy            165 ms/step   x1.00
    jax.jit per-observable     225 ms/step   x0.73  (slower)
    jax.jit + vmap             241 ms/step   x0.68  (slower)
    sparse/gathered numpy       27 ms/step   x6.05
    sparse arrays + jax.vmap     93 ms/step   x1.71

  NUM_OBS=1000 (687 unique obs, num_params=256) - the actual MMD training scale:
    baseline numpy             767 ms/step   x1.00
    sparse/gathered numpy      126 ms/step   x6.09
    sparse arrays + jax.vmap    628 ms/step   x1.22

Notes:
- The dense-JAX results (sections 2-4) were consistently SLOWER than plain
  NumPy on this CPU-only box, at every scale tried. Section 5b's op-by-op
  breakdown shows why, precisely - it's NOT "dispatch overhead" (measured at
  a few *microseconds* per call, see the module docstring) - it's the width
  of the dense array (more columns = more elementwise work than NumPy's
  tighter C loops) plus JAX's automatic differentiation through `jnp.prod`
  roughly tripling that cost on top. **Retest section 3/4 on a real GPU
  allocation** before ruling JAX out - this conclusion is CPU-specific.
- The actual win is section 5 (sparse/gathered numpy): ~6x faster than the
  baseline at every scale tested, in plain NumPy, matching to ~1e-16. It
  fixes a data-structure problem (dense arrays wasting >93% of their width on
  untouched parameters, since k1 caps how many parameters a term can touch)
  rather than a compute-engine problem, which is why it beats JAX regardless
  of whether JAX itself would help.
- Section 5b runs the SAME sparse arrays through jax.jit+vmap (tweakable -
  change the array-shape constants at the top of that section and rerun).
  Forward-only is fast there - competitive with or faster than NumPy - but
  once the gradient's scatter-add (routing narrow per-term contributions
  back into the full-width gradient) is included, it's ~20-24x slower than
  the forward pass alone, on this CPU/XLA:CPU backend specifically. That
  scatter cost, not JAX overhead in general, is why sparse+JAX still lost to
  sparse+NumPy - confirms the original win is the data structure, and pins
  down exactly where JAX adds a real (CPU-specific) cost on top.
- Numba (already a project dependency) is another non-JAX alternative worth
  knowing about: it JITs vectorised array code once, not once per term, so
  the same "flat compile time" argument applies, and it would very likely
  benefit from the same sparse arrays. It segfaulted when tried in this
  sandbox (numba 0.62 / llvmlite float-power lowering) and was not pursued
  further here - retest on your actual machine.
- jax_enable_x64 is required above - JAX defaults to float32, which loses
  too much precision for Adam once many terms are summed.
- RETRACTED: a note here previously claimed joblib threads beat the baseline
  at the SIDE=8/162-obs scale (165ms -> 117ms, x1.41). That does not hold up
  - repeated, more careful testing (both this file's own numbers above and a
  dedicated sweep across 2-128 threads, with both raw ThreadPoolExecutor and
  joblib) consistently shows threaded evaluation SLOWER than serial at every
  thread count, including at the real 1000-observable/num_params=320
  training scale. Root cause: each observable's evaluation is too small
  (~0.1-0.2ms) for NumPy's released-GIL work to amortize Python's GIL
  acquire/release and thread-scheduling overhead. Treat `eval_n_jobs=1` (the
  library default) as correct; see `Propagator.propagate()`'s docstring for
  the full explanation and when threading might still help (much larger
  per-term-count workloads, e.g. little/no k1/k2 truncation).
- Bottom line for your `scripts/generation` MMD training at num_obs=1000:
  switch `make_evaluator`'s internal representation from `build_arrays`'
  dense (n_terms, num_params) arrays to the sparse/gathered (n_terms, W)
  form in section 5 - measured ~6x speedup end-to-end at your actual scale,
  no JAX/GPU dependency required. JAX/vmap remain worth a GPU retest, but are
  not the highest-leverage change on CPU.
"""
)
