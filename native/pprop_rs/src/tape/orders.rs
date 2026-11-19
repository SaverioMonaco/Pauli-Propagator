//! Evaluating a tape with the k2 truncation: every box holds `k2 + 1`
//! numbers instead of one, number `t` being the sum of everything that reached
//! the word with exactly `t` factors (the order).
//!
//! The sum-of-terms engine drops a term with more than k2 factors, where a
//! factor is a sin or cos of a trainable angle (frequency mode) or only a sin
//! (sine mode); fixed angles add none. Here a step moves each contribution up
//! by the factors it adds, and what passes k2 falls off the end:
//!
//! ```text
//! trainable rotation:      new a[t] = c·a[t − sc] − s·b[t − ss]
//!                          new b[t] = s·a[t − ss] + c·b[t − sc]
//!                          (sc, ss) = (1, 1) frequency, (0, 1) sine
//! trainable block entry:   new[out][t] += f · old[in][t − shift]
//!                          shift = n_sin + n_cos frequency, n_sin sine
//! fixed angles, negations: every order on its own (no shift)
//! ```
//!
//! The readout adds up all the orders. With one order (no k2) the tape uses
//! the plain one-number-per-box code in `eval.rs` and `grad.rs` instead.

use super::{Angle, Step, Tape, NONE};

/// How many orders a trainable rotation's cos and sin parts move up.
fn rotation_shifts(angle: &Angle, sine_order: bool) -> (usize, usize) {
    match angle {
        Angle::Param(_) if sine_order => (0, 1),
        Angle::Param(_) => (1, 1),
        Angle::Fixed { .. } => (0, 0),
    }
}

/// How many orders a block entry with `n_sin`, `n_cos` factors moves up.
fn entry_shift(angle: &Angle, sine_order: bool, n_sin: u8, n_cos: u8) -> usize {
    match angle {
        Angle::Param(_) if sine_order => n_sin as usize,
        Angle::Param(_) => (n_sin + n_cos) as usize,
        Angle::Fixed { .. } => 0,
    }
}

/// `x[t - shift]`, or 0 below order 0.
#[inline]
fn shifted(x: &[f64], t: usize, shift: usize) -> f64 {
    if t >= shift { x[t - shift] } else { 0.0 }
}

fn cos_sin(angle: &Angle, sins: &[f64], coss: &[f64]) -> (f64, f64, Option<usize>) {
    match *angle {
        Angle::Param(p) => (coss[p as usize], sins[p as usize], Some(p as usize)),
        Angle::Fixed { cos, sin } => (cos, sin, None),
    }
}

/// Every block entry's `coeff · sⁿ · cᵐ` and its derivative (as in `grad.rs`).
fn block_factors(tape: &Tape, s: f64, c: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut f = Vec::new();
    let mut df = Vec::new();
    for p in &tape.block_progs {
        f.push(p.entries.iter().map(|&(_, _, k, n, m)| k * s.powi(n as i32) * c.powi(m as i32)).collect());
        df.push(
            p.entries
                .iter()
                .map(|&(_, _, k, n, m)| {
                    let (n, m) = (n as i32, m as i32);
                    let mut d = 0.0;
                    if n > 0 {
                        d += n as f64 * s.powi(n - 1) * c.powi(m + 1);
                    }
                    if m > 0 {
                        d -= m as f64 * s.powi(n + 1) * c.powi(m - 1);
                    }
                    k * d
                })
                .collect(),
        );
    }
    (f, df)
}

impl Tape {
    /// The forward pass with `self.orders` numbers per box. With `saved`, the
    /// old values of every box a pair or block writes are pushed onto it (all
    /// orders), for the gradient. Returns the box values.
    fn forward_orders(&self, sins: &[f64], coss: &[f64], mut saved: Option<&mut Vec<f64>>) -> Vec<f64> {
        let k = self.orders();
        let mut v = vec![0.0f64; self.n_boxes as usize * k];
        for &(b, c) in &self.init {
            v[b as usize * k] = c; // a plain coefficient has order 0
        }
        let (mut oa, mut ob) = (vec![0.0; k], vec![0.0; k]);
        let mut old = vec![0.0; 4 * k];
        let mut new = vec![0.0; 4 * k];
        for step in &self.steps {
            match *step {
                Step::Rotate { ref angle, first, last } => {
                    let (c, s, _) = cos_sin(angle, sins, coss);
                    let (sc, ss) = rotation_shifts(angle, self.sine_order);
                    for &[a, b] in &self.pairs[first as usize..last as usize] {
                        let (a, b) = (a as usize * k, b as usize * k);
                        oa.copy_from_slice(&v[a..a + k]);
                        ob.copy_from_slice(&v[b..b + k]);
                        if let Some(sv) = saved.as_deref_mut() {
                            sv.extend_from_slice(&oa);
                            sv.extend_from_slice(&ob);
                        }
                        for t in 0..k {
                            v[a + t] = c * shifted(&oa, t, sc) - s * shifted(&ob, t, ss);
                            v[b + t] = s * shifted(&oa, t, ss) + c * shifted(&ob, t, sc);
                        }
                    }
                }
                Step::Negate { first, last } => {
                    for &b in &self.negs[first as usize..last as usize] {
                        let b = b as usize * k;
                        v[b..b + k].iter_mut().for_each(|x| *x = -*x);
                    }
                }
                Step::Block { ref angle, first, last } => {
                    let (c, s, _) = cos_sin(angle, sins, coss);
                    let (f, _) = block_factors(self, s, c);
                    for blk in &self.blocks[first as usize..last as usize] {
                        let prog = &self.block_progs[blk.prog as usize];
                        let size = prog.size as usize;
                        for m in 0..size {
                            let dst = &mut old[m * k..(m + 1) * k];
                            if blk.members[m] != NONE {
                                let src = blk.members[m] as usize * k;
                                dst.copy_from_slice(&v[src..src + k]);
                            } else {
                                dst.iter_mut().for_each(|x| *x = 0.0);
                            }
                        }
                        if let Some(sv) = saved.as_deref_mut() {
                            sv.extend_from_slice(&old[..size * k]);
                        }
                        new[..size * k].iter_mut().for_each(|x| *x = 0.0);
                        for (&(out, inp, _, ns, nc), fe) in prog.entries.iter().zip(&f[blk.prog as usize]) {
                            let sh = entry_shift(angle, self.sine_order, ns, nc);
                            let (out, inp) = (out as usize * k, inp as usize * k);
                            for t in sh..k {
                                new[out + t] += fe * old[inp + t - sh];
                            }
                        }
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                let dst = blk.members[m] as usize * k;
                                v[dst..dst + k].copy_from_slice(&new[m * k..(m + 1) * k]);
                            }
                        }
                    }
                }
            }
        }
        v
    }

    fn readout_orders(&self, v: &[f64]) -> f64 {
        let k = self.orders();
        self.readout.iter().map(|&b| v[b as usize * k..(b as usize + 1) * k].iter().sum::<f64>()).sum()
    }

    /// The expectation value with the k2 truncation.
    pub(crate) fn value_orders(&self, sins: &[f64], coss: &[f64]) -> f64 {
        self.readout_orders(&self.forward_orders(sins, coss, None))
    }

    /// The expectation value with the k2 truncation, and its gradient in `grad`.
    pub(crate) fn value_and_grad_orders(&self, sins: &[f64], coss: &[f64], grad: &mut [f64]) -> f64 {
        grad.iter_mut().for_each(|g| *g = 0.0);
        let k = self.orders();
        let mut saved = Vec::with_capacity((2 * self.pairs.len() + 4 * self.blocks.len()) * k);
        let v = self.forward_orders(sins, coss, Some(&mut saved));
        let value = self.readout_orders(&v);

        // λ[box][t] = ∂ value / ∂ v[box][t]: 1 for every order of a readout box.
        let mut lam = vec![0.0f64; self.n_boxes as usize * k];
        for &b in &self.readout {
            lam[b as usize * k..(b as usize + 1) * k].iter_mut().for_each(|x| *x += 1.0);
        }
        let (mut la, mut lb) = (vec![0.0; k], vec![0.0; k]);
        let mut lam_out = vec![0.0; 4 * k];
        let mut lam_in = vec![0.0; 4 * k];
        let mut top = saved.len();
        for step in self.steps.iter().rev() {
            match *step {
                Step::Rotate { ref angle, first, last } => {
                    let (c, s, p) = cos_sin(angle, sins, coss);
                    let (sc, ss) = rotation_shifts(angle, self.sine_order);
                    for &[a, b] in self.pairs[first as usize..last as usize].iter().rev() {
                        let (a, b) = (a as usize * k, b as usize * k);
                        top -= 2 * k;
                        let (oa, ob) = (&saved[top..top + k], &saved[top + k..top + 2 * k]);
                        la.copy_from_slice(&lam[a..a + k]);
                        lb.copy_from_slice(&lam[b..b + k]);
                        if let Some(p) = p {
                            for t in 0..k {
                                grad[p] += la[t] * (-s * shifted(oa, t, sc) - c * shifted(ob, t, ss))
                                    + lb[t] * (c * shifted(oa, t, ss) - s * shifted(ob, t, sc));
                            }
                        }
                        // new a[t] reads a[t − sc], b[t − ss]; new b[t] reads a[t − ss], b[t − sc].
                        for u in 0..k {
                            let up = |x: &[f64], sh: usize| if u + sh < k { x[u + sh] } else { 0.0 };
                            lam[a + u] = c * up(&la, sc) + s * up(&lb, ss);
                            lam[b + u] = -s * up(&la, ss) + c * up(&lb, sc);
                        }
                    }
                }
                Step::Negate { first, last } => {
                    for &b in &self.negs[first as usize..last as usize] {
                        let b = b as usize * k;
                        lam[b..b + k].iter_mut().for_each(|x| *x = -*x);
                    }
                }
                Step::Block { ref angle, first, last } => {
                    let (c, s, p) = cos_sin(angle, sins, coss);
                    let (f, df) = block_factors(self, s, c);
                    for blk in self.blocks[first as usize..last as usize].iter().rev() {
                        let prog = &self.block_progs[blk.prog as usize];
                        let size = prog.size as usize;
                        top -= size * k;
                        let old = &saved[top..top + size * k];
                        for m in 0..size {
                            let dst = &mut lam_out[m * k..(m + 1) * k];
                            if blk.members[m] != NONE {
                                let src = blk.members[m] as usize * k;
                                dst.copy_from_slice(&lam[src..src + k]);
                            } else {
                                dst.iter_mut().for_each(|x| *x = 0.0);
                            }
                        }
                        lam_in[..size * k].iter_mut().for_each(|x| *x = 0.0);
                        let (fp, dfp) = (&f[blk.prog as usize], &df[blk.prog as usize]);
                        for (e, &(out, inp, _, ns, nc)) in prog.entries.iter().enumerate() {
                            let sh = entry_shift(angle, self.sine_order, ns, nc);
                            let (out, inp) = (out as usize * k, inp as usize * k);
                            for t in sh..k {
                                if let Some(p) = p {
                                    grad[p] += lam_out[out + t] * dfp[e] * old[inp + t - sh];
                                }
                                lam_in[inp + t - sh] += fp[e] * lam_out[out + t];
                            }
                        }
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                let dst = blk.members[m] as usize * k;
                                lam[dst..dst + k].copy_from_slice(&lam_in[m * k..(m + 1) * k]);
                            }
                        }
                    }
                }
            }
        }
        debug_assert_eq!(top, 0);
        value
    }
}
