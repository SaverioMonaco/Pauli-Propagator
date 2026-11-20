//! What one gate does to one Pauli word and its terms, in the sum-of-terms
//! engine: each evolver emits the words the gate turns it into.

use super::arena::*;
use crate::gates::*;
use crate::pauli::*;

// Output of evolving words through one gate, before truncation and insertion.
pub(crate) type Emitted<const NW: usize> = Vec<(PauliKey<NW>, Vec<Term>)>;

/// Multiply every term's coefficient by a constant, in place.
pub(crate) fn scale_terms(mut terms: Vec<Term>, factor: f64) -> Vec<Term> {
    if factor != 1.0 {
        for t in terms.iter_mut() {
            *t = t.scaled(factor);
        }
    }
    terms
}

// ---------------------------------------------------------------------
// Generic per-gate-shape evolvers
// ---------------------------------------------------------------------
//
// Each evolver takes one Pauli word with its terms and emits the words it
// becomes. The propagation loop only passes in words the gate acts on (see
// `gate_acts`), but the pass-through arms stay so every evolver remains a
// complete transcription of its rule table for `evolve_single_gate_debug`.

/// RX/RY/RZ shape: commuting Pauli passes through; anti-commuting Pauli
/// splits into a cos(theta) branch (same word) and a sign*sin(theta) branch
/// (new word). With a fixed angle the two factors are constants folded into
/// the coefficients, and a branch whose factor is zero is not created.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evolve_rotation<const NW: usize>(
    x: [u64; NW], z: [u64; NW], mut terms: Vec<Term>, wire: u32, param: i64, fixed: Option<f64>,
    rule: fn(u8) -> Option<(u8, i8)>,
    mono: &mut Monomials,
    out: &mut Emitted<NW>,
) {
    let label = pauli_label(&x, &z, wire);
    match rule(label) {
        None => out.push(((x, z), terms)),
        Some((out_label, sign)) => {
            let new_key = set_label(x, z, wire, out_label);
            if let Some(theta) = fixed {
                let (s, c) = fixed_sin_cos(theta);
                let s = sign as f64 * s;
                if c != 0.0 && s != 0.0 {
                    let cos_terms = scale_terms(terms.clone(), c);
                    out.push(((x, z), cos_terms));
                    out.push((new_key, scale_terms(terms, s)));
                } else if c != 0.0 {
                    out.push(((x, z), scale_terms(terms, c)));
                } else if s != 0.0 {
                    out.push((new_key, scale_terms(terms, s)));
                }
                // sin and cos cannot both snap to zero; nothing to do otherwise.
                return;
            }
            let p = param as u32;
            let cos_terms: Vec<Term> = terms.iter().map(|t| t.times(mono, p, false)).collect();
            for t in terms.iter_mut() {
                *t = t.times(mono, p, true).scaled(sign as f64);
            }
            out.push(((x, z), cos_terms));
            out.push((new_key, terms));
        }
    }
}

/// Multiply every coefficient by -1 when `sign` is negative.
pub(crate) fn apply_sign(mut terms: Vec<Term>, sign: i8) -> Vec<Term> {
    if sign != 1 {
        for t in terms.iter_mut() {
            t.coeff = -t.coeff;
        }
    }
    terms
}

/// H/S/SX shape: commuting Pauli passes through; otherwise exactly one
/// output word with a constant +-1 sign (no trig, no branching).
pub(crate) fn evolve_clifford1q<const NW: usize>(
    x: [u64; NW], z: [u64; NW], terms: Vec<Term>, wire: u32,
    rule: fn(u8) -> Option<(u8, i8)>,
    out: &mut Emitted<NW>,
) {
    let label = pauli_label(&x, &z, wire);
    match rule(label) {
        None => out.push(((x, z), terms)),
        Some((out_label, sign)) => {
            let new_key = set_label(x, z, wire, out_label);
            out.push((new_key, apply_sign(terms, sign)));
        }
    }
}

/// T-gate shape: commuting Pauli passes through; otherwise splits into two
/// output words with constant (non-trig) phase multipliers.
pub(crate) fn evolve_t<const NW: usize>(
    x: [u64; NW], z: [u64; NW], terms: Vec<Term>, wire: u32,
    out: &mut Emitted<NW>,
) {
    let label = pauli_label(&x, &z, wire);
    match t_rule(label) {
        None => out.push(((x, z), terms)),
        Some([(l1, ph1), (l2, ph2)]) => {
            let k1 = set_label(x, z, wire, l1);
            let k2 = set_label(x, z, wire, l2);
            let t1: Vec<Term> = terms.iter().map(|t| t.scaled(ph1)).collect();
            let t2: Vec<Term> = terms.into_iter().map(|t| t.scaled(ph2)).collect();
            out.push((k1, t1));
            out.push((k2, t2));
        }
    }
}

pub(crate) fn evolve_swap<const NW: usize>(
    x: [u64; NW], z: [u64; NW], terms: Vec<Term>, w0: u32, w1: u32,
    out: &mut Emitted<NW>,
) {
    let l0 = pauli_label(&x, &z, w0);
    let l1 = pauli_label(&x, &z, w1);
    if l0 == l1 {
        out.push(((x, z), terms));
        return;
    }
    let (nx, nz) = set_label(x, z, w0, l1);
    let (nx, nz) = set_label(nx, nz, w1, l0);
    out.push(((nx, nz), terms));
}

/// CNOT/CY/CZ shape: commuting word passes through; otherwise exactly one
/// output word with a constant +-1 sign applied to every term.
pub(crate) fn evolve_controlled<const NW: usize>(
    x: [u64; NW], z: [u64; NW], terms: Vec<Term>, control: u32, target: u32,
    rule: fn(u8) -> Option<(u8, u8, i8)>,
    out: &mut Emitted<NW>,
) {
    let lc = pauli_label(&x, &z, control);
    let lt = pauli_label(&x, &z, target);
    let code = (lc << 2) | lt;
    match rule(code) {
        None => out.push(((x, z), terms)),
        Some((out_c, out_t, sign)) => {
            let (nx, nz) = set_label(x, z, control, out_c);
            let (nx, nz) = set_label(nx, nz, target, out_t);
            out.push(((nx, nz), apply_sign(terms, sign)));
        }
    }
}

/// CRX/CRY/CRZ shape: commuting word passes through; otherwise up to 4
/// output words, each scaling every existing term by a constant coefficient
/// and appending 0-2 copies of the gate's parameter index to sin_idx/cos_idx.
/// With a fixed angle those sin/cos powers are evaluated at the *half* angle
/// (the rule tables are written in theta/2, see `controlledrotation.py`) and
/// folded into the coefficient; branches whose factor is zero are dropped.
#[allow(clippy::too_many_arguments)]
pub(crate) fn evolve_controlled_rotation<const NW: usize>(
    x: [u64; NW], z: [u64; NW], terms: Vec<Term>, control: u32, target: u32, param: i64, fixed: Option<f64>,
    rule: fn(u8) -> Option<CRRule>,
    mono: &mut Monomials,
    out: &mut Emitted<NW>,
) {
    let lc = pauli_label(&x, &z, control);
    let lt = pauli_label(&x, &z, target);
    let code = (lc << 2) | lt;
    match rule(code) {
        None => out.push(((x, z), terms)),
        Some(branches) => {
            if let Some(theta) = fixed {
                let (s, c) = fixed_sin_cos(theta / 2.0);
                for &(out_c, out_t, coeff, n_sin, n_cos) in branches {
                    let factor = coeff * s.powi(n_sin as i32) * c.powi(n_cos as i32);
                    if factor == 0.0 {
                        continue;
                    }
                    let (nx, nz) = set_label(x, z, control, out_c);
                    let (nx, nz) = set_label(nx, nz, target, out_t);
                    out.push(((nx, nz), scale_terms(terms.clone(), factor)));
                }
                return;
            }
            let p = param as u32;
            for &(out_c, out_t, coeff, n_sin, n_cos) in branches {
                let (nx, nz) = set_label(x, z, control, out_c);
                let (nx, nz) = set_label(nx, nz, target, out_t);
                let new_terms: Vec<Term> = terms
                    .iter()
                    .map(|&t| {
                        let mut t = t.scaled(coeff);
                        for _ in 0..n_sin {
                            t = t.times(mono, p, true);
                        }
                        for _ in 0..n_cos {
                            t = t.times(mono, p, false);
                        }
                        t
                    })
                    .collect();
                out.push(((nx, nz), new_terms));
            }
        }
    }
}

pub(crate) fn evolve_one_gate<const NW: usize>(
    gate: &GateSpec<NW>, x: [u64; NW], z: [u64; NW], terms: Vec<Term>,
    mono: &mut Monomials, out: &mut Emitted<NW>,
) {
    match gate.kind {
        RX => evolve_rotation(x, z, terms, gate.wire0, gate.param, gate.fixed, rx_rule, mono, out),
        RY => evolve_rotation(x, z, terms, gate.wire0, gate.param, gate.fixed, ry_rule, mono, out),
        RZ => evolve_rotation(x, z, terms, gate.wire0, gate.param, gate.fixed, rz_rule, mono, out),
        H => evolve_clifford1q(x, z, terms, gate.wire0, h_rule, out),
        S => evolve_clifford1q(x, z, terms, gate.wire0, s_rule, out),
        SX => evolve_clifford1q(x, z, terms, gate.wire0, sx_rule, out),
        T => evolve_t(x, z, terms, gate.wire0, out),
        SWAP => evolve_swap(x, z, terms, gate.wire0, gate.wire1 as u32, out),
        CNOT => evolve_controlled(x, z, terms, gate.wire0, gate.wire1 as u32, cnot_rule, out),
        CY => evolve_controlled(x, z, terms, gate.wire0, gate.wire1 as u32, cy_rule, out),
        CZ => evolve_controlled(x, z, terms, gate.wire0, gate.wire1 as u32, cz_rule, out),
        CRX => evolve_controlled_rotation(x, z, terms, gate.wire0, gate.wire1 as u32, gate.param, gate.fixed, crx_rule, mono, out),
        CRY => evolve_controlled_rotation(x, z, terms, gate.wire0, gate.wire1 as u32, gate.param, gate.fixed, cry_rule, mono, out),
        CRZ => evolve_controlled_rotation(x, z, terms, gate.wire0, gate.wire1 as u32, gate.param, gate.fixed, crz_rule, mono, out),
        _ => unreachable!("unknown gate kind {}", gate.kind),
    }
}
