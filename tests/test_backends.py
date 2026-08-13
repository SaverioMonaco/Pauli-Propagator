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
from collections import Counter

import numpy as np
import pennylane as qml
import pytest

from pprop import Propagator  # noqa
from pprop.propagator.binding import Free
from pprop.propagator.utils import build_ragged_arrays

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


def test_fixed_value_gates_agree_with_qml():
    """
    A fixed-value gate gets a hidden slot rather than a trainable index, and
    the ragged layout indexes that slot out of the same lookup table as every
    other factor. Value and gradient must still match param-shift, and the
    gradient must carry one column per *trainable* parameter, not per slot.
    """
    def ansatz(params):
        for layer in range(2):
            for qubit in range(num_qubits):
                qml.RX(params[6 * layer + 2 * qubit], wires=qubit)
                qml.RY(params[6 * layer + 2 * qubit + 1], wires=qubit)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 0])
            if layer == 0:
                qml.RY(np.pi, wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

    prop = Propagator(ansatz)
    prop.propagate()
    assert prop._fixed_value_slots  # RY(pi) got a hidden slot
    assert prop._internal_num_params > prop.num_params

    # The hidden slot is a real entry in the ragged table, so the expression
    # references more parameters than the caller ever passes in.
    _, idx, _ = build_ragged_arrays(prop.exprs[0], prop._internal_num_params)
    assert idx.size

    qnode = qml.QNode(ansatz, qml.device("default.qubit", wires=num_qubits))
    for _ in range(3):
        params = qml.numpy.random.uniform(-np.pi, np.pi, prop.num_params)
        val, grad = prop.eval_and_grad(params)
        assert np.allclose(val, qnode(params), atol=1e-6)
        assert np.allclose(grad, qml.gradients.param_shift(qnode)(params), atol=1e-6)
        assert np.asarray(grad).shape[-1] == prop.num_params


def test_gradient_at_zeros_of_sin_and_cos():
    """
    The gradient of a term is normally formed as term * cot(theta) (and the
    tan analogue for the cosine factors), which is singular wherever
    sin(theta) or cos(theta) vanishes. Angles that land in such a zero take an
    exact fallback path instead, so they must still match param-shift - these
    are exactly the values an optimiser tends to walk into.
    """
    def ansatz(params):
        qml.Hadamard(wires=0)
        qml.RX(params[0], wires=0)
        qml.CNOT(wires=[0, 1])
        qml.RY(params[1], wires=1)
        qml.RZ(params[2], wires=0)
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1))

    prop = Propagator(ansatz)
    prop.propagate()
    qnode = qml.QNode(ansatz, qml.device("default.qubit", wires=2))

    for values in ([0.0, 0.0, 0.0],
                   [np.pi / 2, np.pi / 2, np.pi / 2],
                   [np.pi, np.pi / 2, 0.0],
                   [0.0, np.pi, 1e-12]):
        params = qml.numpy.array(values, requires_grad=True)
        val, grad = prop.eval_and_grad(np.asarray(values))
        assert np.allclose(val, qnode(params), atol=1e-6)
        assert np.allclose(grad, qml.gradients.param_shift(qnode)(params), atol=1e-6), (
            f"Mismatch GRAD vs qml at params={values}"
        )


def test_repeated_parameter_powers_agree_with_qml():
    """
    When one trainable index drives several gates, a Pauli path can pick up
    sin(theta_k) or cos(theta_k) more than once, giving powers above 1 - which
    the evaluator carries as repeated factors rather than as an exponent. Both
    the value and the derivative (p * sin^(p-1) * cos, including at the zeros
    where the exact path takes over) have to come out right.
    """
    def ansatz(params):
        for layer in range(3):
            for qubit in range(num_qubits):
                qml.RX(params[qubit % 2], wires=qubit)
                qml.RY(params[2 + (layer + qubit) % 2], wires=qubit)
            qml.CNOT(wires=[0, 1])
            qml.CNOT(wires=[1, 2])
            qml.CNOT(wires=[2, 0])
        return qml.expval(qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2))

    prop = Propagator(ansatz)
    prop.propagate()
    assert prop.num_params == 4

    powers = Counter()
    for _, sin_idx, cos_idx in prop.exprs[0]:
        powers.update(Counter(sin_idx).values())
        powers.update(Counter(cos_idx).values())
    assert max(powers) > 1, "expected this circuit to produce powers above 1"

    qnode = qml.QNode(ansatz, qml.device("default.qubit", wires=num_qubits))
    cases = [[0.7, -1.3, 0.2, 2.1], [0.0, 0.0, 0.0, 0.0],
             [np.pi / 2, np.pi, 0.0, np.pi / 2]]
    for values in cases:
        params = qml.numpy.array(values, requires_grad=True)
        val, grad = prop.eval_and_grad(np.asarray(values))
        assert np.allclose(val, qnode(params), atol=1e-6)
        assert np.allclose(grad, qml.gradients.param_shift(qnode)(params), atol=1e-6), (
            f"Mismatch GRAD vs qml at params={values}"
        )


# %%
test_propagator_agrees_with_qml()
test_eval_n_jobs_matches_single_threaded()
test_fixed_value_parameter_is_not_aliased()
test_controlled_rotations_agree_with_qml()
test_bind_affine_reparametrisation_agrees_with_qml()
test_propagator_beyond_64_qubits()
test_fixed_value_gates_agree_with_qml()
test_gradient_at_zeros_of_sin_and_cos()
test_repeated_parameter_powers_agree_with_qml()


def _uneven_width_ansatz(params):
    """Four qubits, RY+RZ per layer: term supports come out very uneven."""
    k = 0
    for qubit in range(4):
        qml.Hadamard(wires=qubit)
    for _ in range(3):
        for qubit in range(4):
            qml.RY(params[k], wires=qubit); k += 1
        for qubit in range(4):
            qml.RZ(params[k], wires=qubit); k += 1
        for qubit in range(3):
            qml.CNOT(wires=[qubit, qubit + 1])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(3))


def test_uneven_support_widths_agree_with_qml():
    """
    The ragged layout exists because term supports are uneven: the padded
    layout sized every term to the widest one, so the rest carried slots that
    only ever multiply by 1. This circuit spreads its supports across a wide
    range, which is the case that used to waste the most, so pin it against
    param-shift.
    """
    prop = Propagator(_uneven_width_ansatz)
    prop.propagate()

    widths = Counter(len(set(s)) + len(set(c)) for _, s, c in prop.exprs[0])
    assert len(widths) > 4, f"expected a spread of support widths, got {dict(widths)}"
    # The padded layout would have sized every term to the widest.
    assert min(widths) * 2 < max(widths), dict(widths)

    qnode = qml.QNode(_uneven_width_ansatz, qml.device("default.qubit", wires=4))
    rng = np.random.default_rng(0)
    for _ in range(3):
        values = rng.uniform(-np.pi, np.pi, prop.num_params)
        params = qml.numpy.array(values, requires_grad=True)
        val, grad = prop.eval_and_grad(np.asarray(values))
        assert np.allclose(val, qnode(params), atol=1e-6)
        assert np.allclose(grad, qml.gradients.param_shift(qnode)(params), atol=1e-6)


def test_truncation_can_empty_an_expression():
    """
    k1 can discard every term an observable has, leaving nothing to evaluate.
    The ragged evaluator special-cases that (np.multiply.reduceat has no empty
    case), and it has to keep returning a correctly shaped zero gradient
    rather than raising.
    """
    prop = Propagator(_uneven_width_ansatz, k1=2)
    prop.propagate()
    assert len(prop.exprs[0]) == 0, "expected k1=2 to discard everything here"

    params = np.random.default_rng(0).uniform(-np.pi, np.pi, prop.num_params)
    assert prop(params)[0] == 0.0

    val, grad = prop.eval_and_grad(params)
    assert val[0] == 0.0
    assert grad.shape == (1, prop.num_params)
    assert np.all(grad == 0.0)


test_uneven_support_widths_agree_with_qml()
test_truncation_can_empty_an_expression()
