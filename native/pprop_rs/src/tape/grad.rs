//! The gradient of a tape: the forward pass of `value`, saving what every step
//! overwrites, then one pass backwards over the steps (reverse-mode
//! differentiation).
//!
//! Going backwards, every box holds an adjoint `λ` = ∂(value)/∂(box). It
//! starts at 1 for the readout boxes and 0 for the others. A step turns the
//! adjoints of the values it wrote into the adjoints of the values it read,
//! and adds to the gradient of its angle. For a rotation pair:
//!
//! ```text
//! forward:   a' = c·a − s·b              b' = s·a + c·b
//! backward:  grad += λa'·(−s·a − c·b) + λb'·(c·a − s·b)      (d c = −s, d s = c)
//!            λa = c·λa' + s·λb'          λb = −s·λa' + c·λb'
//! ```
//!
//! The backward formulas need the old `a` and `b`, which the in-place forward
//! pass overwrote: that is what the saved values are for, one per pair member
//! and per block member. A negation needs none (`λ` just flips sign), and a
//! fixed angle adds nothing to the gradient.

use super::{Angle, Step, Tape, NONE};

/// `coeff · sⁿ · cᵐ` and its derivative in the angle, for every entry of
/// every kind of block (ds = c, dc = −s).
fn block_factors(tape: &Tape, s: f64, c: f64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut f = Vec::with_capacity(tape.block_progs.len());
    let mut df = Vec::with_capacity(tape.block_progs.len());
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

fn cos_sin(angle: &Angle, sins: &[f64], coss: &[f64]) -> (f64, f64, Option<usize>) {
    match *angle {
        Angle::Param(p) => (coss[p as usize], sins[p as usize], Some(p as usize)),
        Angle::Fixed { cos, sin } => (cos, sin, None),
    }
}

impl Tape {
    /// The expectation value at the given `sin(θ)` and `cos(θ)`, and its
    /// gradient, written into `grad` (zeroed first; as long as the parameter
    /// vector).
    pub(crate) fn value_and_grad(&self, sins: &[f64], coss: &[f64], grad: &mut [f64]) -> f64 {
        if self.k2 >= 0 {
            return self.value_and_grad_orders(sins, coss, grad);
        }
        grad.iter_mut().for_each(|g| *g = 0.0);
        let mut v = vec![0.0f64; self.n_boxes as usize];
        for &(b, c) in &self.init {
            v[b as usize] = c;
        }

        // Forward, saving the old value of every box a pair or block writes.
        let mut saved: Vec<f64> = Vec::with_capacity(2 * self.pairs.len() + self.singles.len() + 4 * self.blocks.len());
        for step in &self.steps {
            match *step {
                Step::Rotate { ref angle, first, last, sfirst, slast } => {
                    let (c, s, _) = cos_sin(angle, sins, coss);
                    for &[a, b] in &self.pairs[first as usize..last as usize] {
                        let (a, b) = (a as usize, b as usize);
                        let (va, vb) = (v[a], v[b]);
                        saved.push(va);
                        saved.push(vb);
                        v[a] = c * va - s * vb;
                        v[b] = s * va + c * vb;
                    }
                    for &b in &self.singles[sfirst as usize..slast as usize] {
                        saved.push(v[b as usize]);
                        v[b as usize] *= c;
                    }
                }
                Step::Negate { first, last } => {
                    for &b in &self.negs[first as usize..last as usize] {
                        v[b as usize] = -v[b as usize];
                    }
                }
                Step::Block { ref angle, first, last } => {
                    let (c, s, _) = cos_sin(angle, sins, coss);
                    let (f, _) = block_factors(self, s, c);
                    for blk in &self.blocks[first as usize..last as usize] {
                        let prog = &self.block_progs[blk.prog as usize];
                        let size = prog.size as usize;
                        let mut old = [0.0f64; 4];
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                old[m] = v[blk.members[m] as usize];
                            }
                            saved.push(old[m]);
                        }
                        let mut new = [0.0f64; 4];
                        for (&(out, inp, ..), fe) in prog.entries.iter().zip(&f[blk.prog as usize]) {
                            new[out as usize] += fe * old[inp as usize];
                        }
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                v[blk.members[m] as usize] = new[m];
                            }
                        }
                    }
                }
            }
        }
        let value = self.readout.iter().map(|&b| v[b as usize]).sum();

        // Backward. `saved` is read from the end, in exactly the reverse order.
        let mut lam = vec![0.0f64; self.n_boxes as usize];
        for &b in &self.readout {
            lam[b as usize] += 1.0;
        }
        let mut top = saved.len();
        for step in self.steps.iter().rev() {
            match *step {
                Step::Rotate { ref angle, first, last, sfirst, slast } => {
                    let (c, s, p) = cos_sin(angle, sins, coss);
                    // A single: a' = c·a, so grad += λa'·(−s·a) and λa = c·λa'.
                    for &b in self.singles[sfirst as usize..slast as usize].iter().rev() {
                        top -= 1;
                        let b = b as usize;
                        if let Some(p) = p {
                            grad[p] += lam[b] * (-s * saved[top]);
                        }
                        lam[b] *= c;
                    }
                    for &[a, b] in self.pairs[first as usize..last as usize].iter().rev() {
                        let (a, b) = (a as usize, b as usize);
                        top -= 2;
                        let (va, vb) = (saved[top], saved[top + 1]);
                        let (la, lb) = (lam[a], lam[b]);
                        if let Some(p) = p {
                            grad[p] += la * (-s * va - c * vb) + lb * (c * va - s * vb);
                        }
                        lam[a] = c * la + s * lb;
                        lam[b] = -s * la + c * lb;
                    }
                }
                Step::Negate { first, last } => {
                    for &b in &self.negs[first as usize..last as usize] {
                        lam[b as usize] = -lam[b as usize];
                    }
                }
                Step::Block { ref angle, first, last } => {
                    let (c, s, p) = cos_sin(angle, sins, coss);
                    let (f, df) = block_factors(self, s, c);
                    for blk in self.blocks[first as usize..last as usize].iter().rev() {
                        let prog = &self.block_progs[blk.prog as usize];
                        let size = prog.size as usize;
                        top -= size;
                        let old = &saved[top..top + size];
                        let mut lam_out = [0.0f64; 4];
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                lam_out[m] = lam[blk.members[m] as usize];
                            }
                        }
                        let mut lam_in = [0.0f64; 4];
                        let (fp, dfp) = (&f[blk.prog as usize], &df[blk.prog as usize]);
                        for (e, &(out, inp, ..)) in prog.entries.iter().enumerate() {
                            let (out, inp) = (out as usize, inp as usize);
                            if let Some(p) = p {
                                grad[p] += lam_out[out] * dfp[e] * old[inp];
                            }
                            lam_in[inp] += fp[e] * lam_out[out];
                        }
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                lam[blk.members[m] as usize] = lam_in[m];
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

#[cfg(test)]
mod tests {
    use super::super::{Block, BlockProg};
    use super::*;

    /// Central finite differences of `value` in every parameter.
    fn numeric_grad(t: &Tape, theta: &[f64]) -> Vec<f64> {
        let eval = |th: &[f64]| {
            let s: Vec<f64> = th.iter().map(|x| x.sin()).collect();
            let c: Vec<f64> = th.iter().map(|x| x.cos()).collect();
            t.value(&s, &c)
        };
        (0..theta.len())
            .map(|k| {
                let (mut up, mut dn) = (theta.to_vec(), theta.to_vec());
                up[k] += 1e-6;
                dn[k] -= 1e-6;
                (eval(&up) - eval(&dn)) / 2e-6
            })
            .collect()
    }

    #[test]
    fn gradient_matches_finite_differences() {
        // Boxes 0..4; a rotation, a negation, a fixed rotation, a 4-member block
        // with squares (CRX's group), and a 2-member block with a NONE member.
        let mut entries = Vec::new();
        let label = |c: u8, t: u8| match (c, t) { (0, 2) => 0, (0, 3) => 1, (2, 2) => 2, (2, 3) => 3, _ => unreachable!() };
        for (word, code) in [(0u8, 2u8), (1, 3), (2, 10), (3, 11)] {
            for &(oc, ot, k, ns, nc) in crate::gates::crx_rule(code).unwrap() {
                entries.push((label(oc, ot), word, k, ns, nc));
            }
        }
        let quad = BlockProg { size: 4, entries };
        let pair = BlockProg { size: 2, entries: vec![(0, 0, 1.0, 0, 1), (0, 1, -1.0, 1, 0), (1, 0, 1.0, 1, 0), (1, 1, 1.0, 0, 1)] };
        let t = Tape {
            n_boxes: 5,
            init: vec![(0, 0.7), (1, -0.2), (4, 0.5)],
            steps: vec![
                Step::Rotate { angle: Angle::Param(0), first: 0, last: 2, sfirst: 0, slast: 0 },
                Step::Negate { first: 0, last: 1 },
                Step::Rotate { angle: Angle::Fixed { cos: 0.6, sin: 0.8 }, first: 2, last: 3, sfirst: 0, slast: 1 },
                Step::Block { angle: Angle::Param(1), first: 0, last: 1 },
                Step::Block { angle: Angle::Param(2), first: 1, last: 2 },
                Step::Rotate { angle: Angle::Param(0), first: 3, last: 4, sfirst: 1, slast: 2 },
            ],
            pairs: vec![[0, 2], [1, 3], [2, 4], [4, 0]],
            negs: vec![2],
            singles: vec![1, 3],
            blocks: vec![Block { prog: 0, members: [0, 1, 2, 3] }, Block { prog: 1, members: [3, NONE, NONE, NONE] }],
            block_progs: vec![quad, pair],
            readout: vec![0, 3],
            k2: -1,
            sine_order: false,
        };
        let theta: [f64; 3] = [0.4, -1.3, 2.2];
        let s: Vec<f64> = theta.iter().map(|x| x.sin()).collect();
        let c: Vec<f64> = theta.iter().map(|x| x.cos()).collect();
        let mut grad = vec![9.0; 3];
        let value = t.value_and_grad(&s, &c, &mut grad);
        assert!((value - t.value(&s, &c)).abs() < 1e-15);
        for (g, n) in grad.iter().zip(numeric_grad(&t, &theta)) {
            assert!((g - n).abs() < 1e-8, "{g} vs {n}");
        }
    }
}
