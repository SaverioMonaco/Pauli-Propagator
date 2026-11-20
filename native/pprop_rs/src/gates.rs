//! Gates: their kind codes, `GateSpec`, the sin/cos of fixed angles, and the
//! rule tables saying what each gate does to a Pauli word's letters.

use crate::pauli::*;

// Gate kind codes, matching _GATE_KIND in src/pprop/propagator/__init__.py.
pub(crate) const RX: u8 = 0;
pub(crate) const RY: u8 = 1;
pub(crate) const RZ: u8 = 2;
pub(crate) const H: u8 = 3;
pub(crate) const S: u8 = 4;
pub(crate) const SX: u8 = 5;
pub(crate) const T: u8 = 6;
pub(crate) const SWAP: u8 = 7;
pub(crate) const CNOT: u8 = 8;
pub(crate) const CY: u8 = 9;
pub(crate) const CZ: u8 = 10;
pub(crate) const CRX: u8 = 11;
pub(crate) const CRY: u8 = 12;
pub(crate) const CRZ: u8 = 13;

pub(crate) struct GateSpec<const NW: usize> {
    pub(crate) kind: u8,
    pub(crate) wire0: u32,
    pub(crate) wire1: i64, // -1 unless two-qubit
    pub(crate) param: i64, // -1 unless read from a trainable parameter slot
    // Constant angle of a rotation/controlled-rotation gate that was given a
    // plain float rather than a trainable index (e.g. `qml.RY(np.pi, wires=0)`).
    // `None` for trainable and non-parametrised gates. See `fixed_sin_cos`.
    pub(crate) fixed: Option<f64>,
    pub(crate) wire_mask: [u64; NW],
}

impl<const NW: usize> GateSpec<NW> {
    /// One gate from the arrays Python passes (`wire1 = -1` for a one-qubit
    /// gate, `param = -1` unless trainable, `fixed` for a constant angle).
    pub(crate) fn new(kind: u8, wire0: u32, wire1: i64, param: i64, fixed: Option<f64>) -> Self {
        let mut wire_mask = [0u64; NW];
        set_wire_bit(&mut wire_mask, wire0);
        if wire1 >= 0 {
            set_wire_bit(&mut wire_mask, wire1 as u32);
        }
        GateSpec { kind, wire0, wire1, param, fixed, wire_mask }
    }
}

// ---------------------------------------------------------------------
// Fixed-angle folding
// ---------------------------------------------------------------------
//
// A rotation given a constant angle contributes constant factors, not
// trigonometric ones, so - exactly like the T gate's 1/sqrt(2) - they are
// multiplied straight into each term's coefficient instead of being pushed
// onto its sin/cos index lists. Two things follow:
//
//   * a branch whose folded factor is zero is dropped before it is ever
//     created. At any multiple of pi/2 one of sin/cos vanishes, so a fixed
//     RY(pi) or CRX(pi) splits nothing and the term count stays what it
//     would be without the gate;
//   * a fixed gate no longer counts towards the `k2` frequency cutoff, which
//     makes it consistent with T (a constant is a constant).
//
// `FOLD_SNAP` is the one judgement call: `f64::sin(PI)` is 1.22e-16, not 0,
// and without snapping to a true zero the dead branch survives carrying
// float noise. The worst case of snapping is dropping a term of size
// |coeff| * 1e-12 per snapped factor, before accumulation across terms/gates.
// This also snaps genuinely nonzero factors near quarter turns. It is an
// approximation, not a test that an angle is exactly Clifford; large
// observable coefficients amplify its absolute error (see propagate docs).

pub(crate) const FOLD_SNAP: f64 = 1e-12;

#[inline]
pub(crate) fn snap(v: f64) -> f64 {
    if v.abs() < FOLD_SNAP { 0.0 } else { v }
}

/// `(sin(theta), cos(theta))` of a fixed angle, with exact zeros at the
/// quarter turns. Callers pass the angle in whatever convention the gate's
/// rule table is written in (the full angle for RX/RY/RZ, the half angle
/// for CRX/CRY/CRZ - see `controlledrotation.py`).
pub(crate) fn fixed_sin_cos(theta: f64) -> (f64, f64) {
    let (s, c) = theta.sin_cos();
    (snap(s), snap(c))
}

// ---------------------------------------------------------------------
// Rule tables (transcribed from pprop/gates/*.py)
// ---------------------------------------------------------------------

// Single-qubit rotations: RX/RY/RZ. rule(label) -> Some((out_label, sign)) for
// the anti-commuting branch, None if `label` commutes with the axis.
pub(crate) fn rx_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        3 => Some((2, -1)), // Y -> Z
        2 => Some((3, 1)),  // Z -> Y
        _ => None,
    }
}
pub(crate) fn ry_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((2, 1)),  // X -> Z
        2 => Some((1, -1)), // Z -> X
        _ => None,
    }
}
pub(crate) fn rz_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((3, -1)), // X -> Y
        3 => Some((1, 1)),  // Y -> X
        _ => None,
    }
}

// Single-qubit Clifford gates: H/S/SX. rule(label) -> Some((out_label, sign)).
// Always exactly one output branch, since these are non-parametrised and
// don't split into a sin/cos pair the way rotations do.
pub(crate) fn h_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((2, 1)),  // X -> Z
        3 => Some((3, -1)), // Y -> -Y
        2 => Some((1, 1)),  // Z -> X
        _ => None,
    }
}
pub(crate) fn s_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((3, -1)), // X -> -Y
        3 => Some((1, 1)),  // Y -> X
        _ => None,
    }
}
pub(crate) fn sx_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        3 => Some((2, -1)), // Y -> -Z
        2 => Some((3, 1)),  // Z -> Y
        _ => None,
    }
}

// T gate: single-qubit non-Clifford, splits into two branches with constant
// (non-trigonometric) phases.
pub(crate) fn t_rule(label: u8) -> Option<[(u8, f64); 2]> {
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    match label {
        1 => Some([(1, INV_SQRT2), (3, -INV_SQRT2)]),  // X -> +X/sqrt2 - Y/sqrt2
        3 => Some([(3, INV_SQRT2), (1, INV_SQRT2)]),   // Y -> +Y/sqrt2 + X/sqrt2
        _ => None,
    }
}

// Two-qubit non-parametrised controlled gates: CNOT/CY/CZ. rule(code) ->
// Some((out_control, out_target, sign)), code = (control_label << 2) | target_label.
pub(crate) fn cnot_rule(code: u8) -> Option<(u8, u8, i8)> {
    match code {
        3 => Some((2, 3, 1)),   // IY -> ZY
        2 => Some((2, 2, 1)),   // IZ -> ZZ
        4 => Some((1, 1, 1)),   // XI -> XX
        5 => Some((1, 0, 1)),   // XX -> XI
        7 => Some((3, 2, 1)),   // XY -> YZ
        6 => Some((3, 3, -1)),  // XZ -> -YY
        12 => Some((3, 1, 1)),  // YI -> YX
        13 => Some((3, 0, 1)),  // YX -> YI
        15 => Some((1, 2, -1)), // YY -> -XZ
        14 => Some((1, 3, 1)),  // YZ -> XY
        11 => Some((0, 3, 1)),  // ZY -> IY
        10 => Some((0, 2, 1)),  // ZZ -> IZ
        _ => None,
    }
}
pub(crate) fn cy_rule(code: u8) -> Option<(u8, u8, i8)> {
    match code {
        1 => Some((2, 1, 1)),   // IX -> ZX
        2 => Some((2, 2, 1)),   // IZ -> ZZ
        4 => Some((1, 3, 1)),   // XI -> XY
        5 => Some((3, 2, -1)),  // XX -> -YZ
        7 => Some((1, 0, 1)),   // XY -> XI
        6 => Some((3, 1, 1)),   // XZ -> YX
        12 => Some((3, 3, 1)),  // YI -> YY
        13 => Some((1, 2, 1)),  // YX -> XZ
        15 => Some((3, 0, 1)),  // YY -> YI
        14 => Some((1, 1, -1)), // YZ -> -XX
        9 => Some((0, 1, 1)),   // ZX -> IX
        10 => Some((0, 2, 1)),  // ZZ -> IZ
        _ => None,
    }
}
pub(crate) fn cz_rule(code: u8) -> Option<(u8, u8, i8)> {
    match code {
        1 => Some((2, 1, 1)),   // IX -> ZX
        3 => Some((2, 3, 1)),   // IY -> ZY
        4 => Some((1, 2, 1)),   // XI -> XZ
        5 => Some((3, 3, 1)),   // XX -> YY
        7 => Some((3, 1, -1)),  // XY -> -YX
        6 => Some((1, 0, 1)),   // XZ -> XI
        12 => Some((3, 2, 1)),  // YI -> YZ
        13 => Some((1, 3, -1)), // YX -> -XY
        15 => Some((1, 1, 1)),  // YY -> XX
        14 => Some((3, 0, 1)),  // YZ -> YI
        9 => Some((0, 1, 1)),   // ZX -> IX
        11 => Some((0, 3, 1)),  // ZY -> IY
        _ => None,
    }
}

// Controlled rotations: CRX/CRY/CRZ. Each entry is a fixed-size list of
// (out_control, out_target, coeff, n_sin, n_cos). n_sin/n_cos count how many
// times the gate's parameter index gets pushed onto sin_idx/cos_idx (0, 1,
// or 2), matching the cos(t/2)/sin(t/2)/cos^2/sin^2/sin*cos factors in
// pprop/gates/controlledrotation.py.
pub(crate) type CRRule = &'static [(u8, u8, f64, u8, u8)];
pub(crate) fn crx_rule(code: u8) -> Option<CRRule> {
    match code {
        3 => Some(&[(0, 3, 1.0, 0, 2), (0, 2, -1.0, 1, 1), (2, 3, 1.0, 2, 0), (2, 2, 1.0, 1, 1)]), // IY
        2 => Some(&[(0, 2, 1.0, 0, 2), (0, 3, 1.0, 1, 1), (2, 2, 1.0, 2, 0), (2, 3, -1.0, 1, 1)]), // IZ
        4 => Some(&[(1, 0, 1.0, 0, 1), (3, 1, 1.0, 1, 0)]),  // XI
        5 => Some(&[(1, 1, 1.0, 0, 1), (3, 0, 1.0, 1, 0)]),  // XX
        7 => Some(&[(1, 3, 1.0, 0, 1), (1, 2, -1.0, 1, 0)]), // XY
        6 => Some(&[(1, 2, 1.0, 0, 1), (1, 3, 1.0, 1, 0)]),  // XZ
        12 => Some(&[(3, 0, 1.0, 0, 1), (1, 1, -1.0, 1, 0)]), // YI
        13 => Some(&[(3, 1, 1.0, 0, 1), (1, 0, -1.0, 1, 0)]), // YX
        15 => Some(&[(3, 3, 1.0, 0, 1), (3, 2, -1.0, 1, 0)]), // YY
        14 => Some(&[(3, 2, 1.0, 0, 1), (3, 3, 1.0, 1, 0)]),  // YZ
        11 => Some(&[(2, 3, 1.0, 0, 2), (2, 2, -1.0, 1, 1), (0, 3, 1.0, 2, 0), (0, 2, 1.0, 1, 1)]), // ZY
        10 => Some(&[(2, 2, 1.0, 0, 2), (2, 3, 1.0, 1, 1), (0, 2, 1.0, 2, 0), (0, 3, -1.0, 1, 1)]), // ZZ
        _ => None,
    }
}
pub(crate) fn cry_rule(code: u8) -> Option<CRRule> {
    match code {
        1 => Some(&[(0, 1, 1.0, 0, 2), (0, 2, 1.0, 1, 1), (2, 1, 1.0, 2, 0), (2, 2, -1.0, 1, 1)]), // IX
        2 => Some(&[(0, 2, 1.0, 0, 2), (0, 1, -1.0, 1, 1), (2, 2, 1.0, 2, 0), (2, 1, 1.0, 1, 1)]), // IZ
        4 => Some(&[(1, 0, 1.0, 0, 1), (3, 3, 1.0, 1, 0)]),  // XI
        5 => Some(&[(1, 1, 1.0, 0, 1), (1, 2, 1.0, 1, 0)]),  // XX
        7 => Some(&[(1, 3, 1.0, 0, 1), (3, 0, 1.0, 1, 0)]),  // XY
        6 => Some(&[(1, 2, 1.0, 0, 1), (1, 1, -1.0, 1, 0)]), // XZ
        12 => Some(&[(3, 0, 1.0, 0, 1), (1, 3, -1.0, 1, 0)]), // YI
        13 => Some(&[(3, 1, 1.0, 0, 1), (3, 2, 1.0, 1, 0)]),  // YX
        15 => Some(&[(3, 3, 1.0, 0, 1), (1, 0, -1.0, 1, 0)]), // YY
        14 => Some(&[(3, 2, 1.0, 0, 1), (3, 1, -1.0, 1, 0)]), // YZ
        9 => Some(&[(2, 1, 1.0, 0, 2), (2, 2, 1.0, 1, 1), (0, 1, 1.0, 2, 0), (0, 2, -1.0, 1, 1)]), // ZX
        10 => Some(&[(2, 2, 1.0, 0, 2), (2, 1, -1.0, 1, 1), (0, 2, 1.0, 2, 0), (0, 1, 1.0, 1, 1)]), // ZZ
        _ => None,
    }
}
pub(crate) fn crz_rule(code: u8) -> Option<CRRule> {
    match code {
        1 => Some(&[(0, 1, 1.0, 0, 2), (0, 3, -1.0, 1, 1), (2, 1, 1.0, 2, 0), (2, 3, 1.0, 1, 1)]), // IX
        3 => Some(&[(0, 3, 1.0, 0, 2), (0, 1, 1.0, 1, 1), (2, 3, 1.0, 2, 0), (2, 1, -1.0, 1, 1)]), // IY
        4 => Some(&[(1, 0, 1.0, 0, 1), (3, 2, 1.0, 1, 0)]),  // XI
        5 => Some(&[(1, 1, 1.0, 0, 1), (1, 3, -1.0, 1, 0)]), // XX
        7 => Some(&[(1, 3, 1.0, 0, 1), (1, 1, 1.0, 1, 0)]),  // XY
        6 => Some(&[(1, 2, 1.0, 0, 1), (3, 0, 1.0, 1, 0)]),  // XZ
        12 => Some(&[(3, 0, 1.0, 0, 1), (1, 2, -1.0, 1, 0)]), // YI
        13 => Some(&[(3, 1, 1.0, 0, 1), (3, 3, -1.0, 1, 0)]), // YX
        15 => Some(&[(3, 3, 1.0, 0, 1), (3, 1, 1.0, 1, 0)]), // YY
        14 => Some(&[(3, 2, 1.0, 0, 1), (1, 0, -1.0, 1, 0)]), // YZ
        9 => Some(&[(2, 1, 1.0, 0, 2), (2, 3, -1.0, 1, 1), (0, 1, 1.0, 2, 0), (0, 3, 1.0, 1, 1)]), // ZX
        11 => Some(&[(2, 3, 1.0, 0, 2), (2, 1, 1.0, 1, 1), (0, 3, 1.0, 2, 0), (0, 1, -1.0, 1, 1)]), // ZY
        _ => None,
    }
}

/// Whether `gate` changes the word `(x, z)` at all, i.e. whether its rule
/// table has an entry for the word's labels on the gate's wires. Words it
/// does not act on keep their key and terms unchanged.
pub(crate) fn gate_acts<const NW: usize>(gate: &GateSpec<NW>, x: &[u64; NW], z: &[u64; NW]) -> bool {
    let l0 = pauli_label(x, z, gate.wire0);
    let two = || (l0 << 2) | pauli_label(x, z, gate.wire1 as u32);
    match gate.kind {
        RX => rx_rule(l0).is_some(),
        RY => ry_rule(l0).is_some(),
        RZ => rz_rule(l0).is_some(),
        H => h_rule(l0).is_some(),
        S => s_rule(l0).is_some(),
        SX => sx_rule(l0).is_some(),
        T => t_rule(l0).is_some(),
        SWAP => l0 != pauli_label(x, z, gate.wire1 as u32),
        CNOT => cnot_rule(two()).is_some(),
        CY => cy_rule(two()).is_some(),
        CZ => cz_rule(two()).is_some(),
        CRX => crx_rule(two()).is_some(),
        CRY => cry_rule(two()).is_some(),
        CRZ => crz_rule(two()).is_some(),
        _ => unreachable!("unknown gate kind {}", gate.kind),
    }
}
