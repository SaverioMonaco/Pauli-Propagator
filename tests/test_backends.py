# %%
"""
Verifies Propagator's (sole, Rust-backed) propagation + evaluation path
against PennyLane's own param-shift gradient, across the full gate set
(H/S/SX/T, RX/RY/RZ, SWAP, CNOT/CY/CZ). This is what used to be a
three-backend cross-check ("standard"/"sparse"/"vmap") before those were
removed in favour of the single Rust implementation. See
personal/rust_port.tex for why.
"""
import random

import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator  # noqa
from pprop.propagator.binding import Free
from pprop.propagator.utils import build_sparse_arrays

num_qubits = 3

sqnp_gates = [qml.H, qml.S, qml.T, qml.SX]
sqp_gates = [qml.RX, qml.RY, qml.RZ]
tqnp_gates = [qml.CNOT, qml.CY, qml.CZ, qml.SWAP]


# %%
def get_random_ansatz():
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
            qml.expval(13 * qml.PauliZ(2) + qml.PauliZ(0) @ qml.PauliX(1)),
        ]

    return ansatz


# %%
def test_propagator_agrees_with_qml():
    device = qml.device("default.qubit", wires=num_qubits)

    for _ in range(3):
        ansatz = get_random_ansatz()
        qnode = qml.QNode(ansatz, device)

        prop = Propagator(ansatz)
        prop.propagate()

        for _ in range(5):
            random_params = qml.numpy.random.uniform(-np.pi, np.pi, prop.num_params)
            qml_output = qnode(random_params)
            qml_grad = qml.gradients.param_shift(qnode)(random_params)

            prop_output, prop_grad = prop.eval_and_grad(random_params)

            assert np.allclose(prop_output, qml_output, atol=1e-6), (
                f"Mismatch EVAL vs qml:\nprop: {prop_output}\nqml:  {qml_output}"
            )
            assert np.allclose(prop_grad, qml_grad, atol=1e-6), (
                f"Mismatch GRAD vs qml:\nprop: {prop_grad}\nqml:  {qml_grad}"
            )


def test_eval_n_jobs_matches_single_threaded():
    ansatz = get_random_ansatz()

    prop_serial = Propagator(ansatz)
    prop_serial.propagate(eval_n_jobs=1)

    prop_threaded = Propagator(ansatz)
    prop_threaded.propagate(eval_n_jobs=4)

    random_params = qml.numpy.random.uniform(-np.pi, np.pi, prop_serial.num_params)
    vals_serial, grads_serial = prop_serial.eval_and_grad(random_params)
    vals_threaded, grads_threaded = prop_threaded.eval_and_grad(random_params)

    assert np.allclose(vals_serial, vals_threaded, atol=1e-12)
    assert np.allclose(grads_serial, grads_threaded, atol=1e-12)


def test_fixed_value_parameter_is_not_aliased():
    """
    A gate given a plain float parameter (e.g. qml.RY(0.3, wires=q), not
    derived from indexing the trainable params array) is documented as a
    fixed, non-trainable value (see Gate's docstring). Propagator used to
    truncate it via int(...), silently aliasing it onto whatever trainable
    index it happened to floor to; it must instead stay fixed regardless of
    what the caller passes for the real trainable parameters.
    """
    def ansatz(params):
        qml.RY(0.3, wires=0)
        qml.RX(params[0], wires=1)
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(0))

    device = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(ansatz, device)

    prop = Propagator(ansatz)
    prop.propagate()

    assert prop.num_params == 1  # only RX's index counts; RY(0.3) is fixed

    expected = float(qnode(np.array([0.0])))  # independent of the trainable RX
    for value in (0.0, 0.5, 1.0, 2.0, -3.0):
        out = prop(np.array([value]))[0]
        assert np.isclose(out, expected, atol=1e-10), (
            f"RY(0.3) leaked onto the trainable index at params[0]={value}: "
            f"got {out}, expected the fixed {expected}"
        )

    val, grad = prop.eval_and_grad(np.array([0.5]))
    assert grad.shape == (1, 1)  # no gradient column for the fixed RY


def test_controlled_rotations_agree_with_qml():
    """
    CRX/CRY/CRZ use PennyLane's θ/2 convention (see controlledrotation.py):
    the ansatz reads a normal trainable index, and that index's value must
    be halved at eval time (not inside the ansatz) to match PennyLane. This
    checks that convention end-to-end, including gradients, since CRX/CRY/CRZ
    are excluded from get_random_ansatz's cross-check above.
    """
    device = qml.device("default.qubit", wires=3)
    cr_gates = [qml.CRX, qml.CRY, qml.CRZ]

    for _ in range(5):
        layers = []
        param_idx = 0
        half_angle_indices = []
        for _ in range(4):
            gate = random.choice(sqp_gates + [None])
            qubit = random.randrange(num_qubits)
            single = None
            if gate is not None:
                single = (gate, qubit, param_idx)
                param_idx += 1

            cr_gate = random.choice(cr_gates)
            q0, q1 = random.sample(range(num_qubits), 2)
            half_angle_indices.append(param_idx)
            layers.append((single, (cr_gate, q0, q1, param_idx)))
            param_idx += 1

        def ansatz(params, layers=layers):
            for single, (cr_gate, q0, q1, cr_idx) in layers:
                if single is not None:
                    gate, qubit, idx = single
                    gate(params[idx], wires=qubit)
                cr_gate(params[cr_idx], wires=[q0, q1])
            return qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2))

        qnode = qml.QNode(ansatz, device)
        prop = Propagator(ansatz)
        prop.propagate()

        for _ in range(3):
            true_params = qml.numpy.random.uniform(-np.pi, np.pi, prop.num_params)
            qml_output = qnode(true_params)
            qml_grad = qml.gradients.param_shift(qnode)(true_params)

            eval_params = np.array(true_params)
            eval_params[half_angle_indices] /= 2
            prop_output, prop_grad = prop.eval_and_grad(eval_params)

            # Chain rule: prop_grad is d(output)/d(eval_params), and
            # eval_params[i] = true_params[i]/2 at half-angle indices, so
            # d(output)/d(true_params[i]) = prop_grad[i] / 2 there.
            prop_grad_wrt_true = np.array(prop_grad)
            prop_grad_wrt_true[:, half_angle_indices] /= 2

            assert np.allclose(prop_output, qml_output, atol=1e-6), (
                f"Mismatch EVAL vs qml:\nprop: {prop_output}\nqml:  {qml_output}"
            )
            assert np.allclose(prop_grad_wrt_true, qml_grad, atol=1e-6), (
                f"Mismatch GRAD vs qml:\nprop: {prop_grad_wrt_true}\nqml:  {qml_grad}"
            )


def test_bind_affine_reparametrisation_agrees_with_qml():
    """
    Propagator.bind() lets several gates depend affinely on a smaller set of
    free parameters (e.g. one gate reading f0, another -2*f0 + 3*f1 - 0.5).
    This must match a plain PennyLane circuit written directly in terms of
    the free parameters, value and gradient, since it's exact linear algebra
    (a Jacobian multiply) on top of pprop's own exact gradient, not an
    approximation.
    """
    def ansatz(params):
        qml.Hadamard(wires=0)
        qml.RY(params[0], wires=0)   # f0
        qml.RX(params[1], wires=1)   # -2 * f0
        qml.CNOT(wires=[0, 1])
        qml.RZ(params[2], wires=1)   # f1
        qml.RY(params[3], wires=0)   # f0 + 3*f1 - 0.5
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    def free_ansatz(free):
        qml.Hadamard(wires=0)
        qml.RY(free[0], wires=0)
        qml.RX(-2 * free[0], wires=1)
        qml.CNOT(wires=[0, 1])
        qml.RZ(free[1], wires=1)
        qml.RY(free[0] + 3 * free[1] - 0.5, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    prop = Propagator(ansatz)
    prop.propagate()

    f0, f1 = Free.vars(2)
    bound = prop.bind([f0, -2 * f0, f1, f0 + 3 * f1 - 0.5])
    assert bound.num_free == 2

    device = qml.device("default.qubit", wires=2)
    qnode = qml.QNode(free_ansatz, device)

    for _ in range(5):
        free_true = qml.numpy.random.uniform(-2, 2, 2)
        qml_output = qnode(free_true)
        qml_grad = qml.gradients.param_shift(qnode)(free_true)

        prop_output, prop_grad = bound.eval_and_grad(np.asarray(free_true))

        assert np.allclose(prop_output, qml_output, atol=1e-6)
        assert np.allclose(prop_grad, qml_grad, atol=1e-6)
        assert np.allclose(bound(np.asarray(free_true)), prop_output)

    with pytest.raises(ValueError):
        prop.bind([f0, f1])  # wrong length: must match prop.num_params


def test_propagator_beyond_64_qubits():
    """
    pprop_rs packs each Pauli word into a handful of u64 words sized to the
    circuit (see native/pprop_rs/src/lib.rs's `words_needed`/`NW`), not a
    single u64, so it isn't capped at 64 qubits. This can't be checked by
    just comparing against PennyLane's own statevector simulation (its
    ndarray backend caps out at 64 wires), so instead: build one ansatz
    whose real gates/observables sit on a small, low-index block of wires,
    then a second, otherwise-identical ansatz that additionally applies a
    Barrier spanning every wire up to some qubit count well past 64 (a
    circuit no-op, per Propagator.__init__, but it still grows
    Propagator.num_qubits, since that's read off the recorded tape's wire
    set). Both should propagate to the exact same expression, and the first
    can still be checked against PennyLane directly.
    """
    device = qml.device("default.qubit", wires=num_qubits)
    ansatz = get_random_ansatz()

    for total_qubits in (65, 130, 513):
        def padded_ansatz(params, _ansatz=ansatz, _total=total_qubits):
            # Barrier must be queued before any measurements, so pad first.
            qml.Barrier(wires=list(range(num_qubits, _total)))
            return _ansatz(params)

        qnode = qml.QNode(ansatz, device)
        prop_ref = Propagator(ansatz)
        prop_ref.propagate()

        prop_padded = Propagator(padded_ansatz)
        assert prop_padded.num_qubits == total_qubits
        prop_padded.propagate(use_dead_qubit_pruner=True, use_xy_weight_pruner=True)

        random_params = qml.numpy.random.uniform(-np.pi, np.pi, prop_ref.num_params)
        qml_output = qnode(random_params)
        ref_output = prop_ref(random_params)
        padded_output = prop_padded(random_params)

        assert np.allclose(ref_output, qml_output, atol=1e-6)
        assert np.allclose(padded_output, qml_output, atol=1e-6), (
            f"num_qubits={total_qubits}: padded prop {padded_output} != qml {qml_output}"
        )


def _live_slots(arrays):
    """Slots still referenced by a nonzero-power factor (padding carries power 0)."""
    _, idx_sin, pow_sin, idx_cos, pow_cos = arrays
    return set(idx_sin[pow_sin > 0].tolist()) | set(idx_cos[pow_cos > 0].tolist())


def _live_cells(arrays):
    """Total sin/cos factors actually gathered per call."""
    return int((arrays[2] > 0).sum() + (arrays[4] > 0).sum())


def test_fixed_value_gates_are_constant_folded():
    """
    A fixed-value gate contributes a constant sin/cos factor, so it can be
    folded into the coefficients at build time rather than re-evaluated on
    every call. Terms carrying a sin(pi) factor are identically zero for every
    parameter vector and disappear altogether - modulo the ~1e-16 that
    np.sin(np.pi) actually returns, which is what the snap tolerance handles.
    """
    def ansatz(params):
        for q in range(num_qubits):
            qml.RX(params[q], wires=q)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RY(np.pi, wires=0)          # fixed, not trainable
        for q in range(num_qubits):
            qml.RY(params[num_qubits + q], wires=q)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    prop = Propagator(ansatz)
    prop.propagate()
    assert prop._fixed_value_slots, "expected RY(pi) to get a fixed slot"

    expr = prop.exprs[0]
    n_internal = prop._internal_num_params
    fixed_slots = set(prop._fixed_value_slots.values())
    unfolded = build_sparse_arrays(expr, n_internal)
    folded = build_sparse_arrays(expr, n_internal, prop._value_by_slot())

    # The invariant folding buys us isn't "fewer terms" (that only happens for
    # the angles whose sin or cos is zero) but "the fixed slots are gone": no
    # surviving factor may still reference one.
    assert fixed_slots <= _live_slots(unfolded), "test is vacuous: nothing to fold"
    assert not (fixed_slots & _live_slots(folded)), (
        f"fixed slots {sorted(fixed_slots & _live_slots(folded))} survived folding"
    )

    # and the answer is unchanged
    qnode = qml.QNode(ansatz, qml.device("default.qubit", wires=num_qubits))
    for _ in range(3):
        params = qml.numpy.random.uniform(-np.pi, np.pi, prop.num_params)
        val, grad = prop.eval_and_grad(params)
        assert np.allclose(val, qnode(params), atol=1e-6)
        assert np.allclose(grad, qml.gradients.param_shift(qnode)(params), atol=1e-6)


def test_fixed_value_gates_off_the_quarter_turn_grid_are_folded():
    """
    Sibling of test_fixed_value_gates_are_constant_folded for a fixed angle
    that is *not* a multiple of pi/2: sin(0.3) and cos(0.3) are both nonzero,
    so nothing gets snapped to zero and no term is killed. Instead every term
    is rescaled and loses the fixed slot from its sin/cos support - the
    rescale-and-shrink path rather than the drop path.
    """
    def ansatz(params):
        for q in range(num_qubits):
            qml.RX(params[q], wires=q)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.RY(0.3, wires=0)            # fixed, not trainable, not a quarter turn
        for q in range(num_qubits):
            qml.RY(params[num_qubits + q], wires=q)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    prop = Propagator(ansatz)
    prop.propagate()
    assert prop._fixed_value_slots, "expected RY(0.3) to get a fixed slot"

    expr = prop.exprs[0]
    n_internal = prop._internal_num_params
    fixed_slots = set(prop._fixed_value_slots.values())
    unfolded = build_sparse_arrays(expr, n_internal)
    folded = build_sparse_arrays(expr, n_internal, prop._value_by_slot())

    # Generic angle: sin(0.3) and cos(0.3) are both nonzero, so no term dies.
    # Array *widths* are a global max over terms and can stay flat even when
    # every single term shrank, so count the live gather cells instead.
    assert len(folded[0]) == len(unfolded[0]), "no term should be dropped for a generic angle"
    assert not np.allclose(folded[0], unfolded[0]), "coefficients should have been rescaled"
    assert fixed_slots <= _live_slots(unfolded), "test is vacuous: nothing to fold"
    assert not (fixed_slots & _live_slots(folded)), "fixed slot survived folding"
    assert _live_cells(folded) < _live_cells(unfolded), "fewer factors to gather per call"

    # and the answer is unchanged
    qnode = qml.QNode(ansatz, qml.device("default.qubit", wires=num_qubits))
    for _ in range(3):
        params = qml.numpy.random.uniform(-np.pi, np.pi, prop.num_params)
        val, grad = prop.eval_and_grad(params)
        assert np.allclose(val, qnode(params), atol=1e-6)
        assert np.allclose(grad, qml.gradients.param_shift(qnode)(params), atol=1e-6)


# %%
test_propagator_agrees_with_qml()
test_eval_n_jobs_matches_single_threaded()
test_fixed_value_parameter_is_not_aliased()
test_controlled_rotations_agree_with_qml()
test_bind_affine_reparametrisation_agrees_with_qml()
test_propagator_beyond_64_qubits()
test_fixed_value_gates_are_constant_folded()
test_fixed_value_gates_off_the_quarter_turn_grid_are_folded()
