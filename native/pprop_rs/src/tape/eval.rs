//! Evaluating a tape: one pass over its steps, on one array of box values.

use super::{Angle, Step, Tape, NONE};

impl Tape {
    /// The expectation value at the given `sin(θ)` and `cos(θ)`.
    ///
    /// Every box starts at 0 except the observable's. A rotation step mixes its
    /// pairs; no box appears twice in one step (the compiler's stamps make
    /// sure of that), so a pair only reads and writes its own two boxes and
    /// can be updated in place. A negate step flips the sign of its boxes. A
    /// block step mixes each block's members by its `BlockProg`; the blocks of
    /// one step share no box either.
    pub(crate) fn value(&self, sins: &[f64], coss: &[f64]) -> f64 {
        if self.k2 >= 0 {
            return self.value_orders(sins, coss);
        }
        let mut v = vec![0.0f64; self.n_boxes as usize];
        for &(b, c) in &self.init {
            v[b as usize] = c;
        }
        for step in &self.steps {
            match *step {
                Step::Rotate { ref angle, first, last } => {
                    let (c, s) = match *angle {
                        Angle::Param(p) => (coss[p as usize], sins[p as usize]),
                        Angle::Fixed { cos, sin } => (cos, sin),
                    };
                    for &[a, b] in &self.pairs[first as usize..last as usize] {
                        let (a, b) = (a as usize, b as usize);
                        let (va, vb) = (v[a], v[b]);
                        v[a] = c * va - s * vb;
                        v[b] = s * va + c * vb;
                    }
                }
                Step::Negate { first, last } => {
                    for &b in &self.negs[first as usize..last as usize] {
                        v[b as usize] = -v[b as usize];
                    }
                }
                Step::Block { ref angle, first, last } => {
                    let (c, s) = match *angle {
                        Angle::Param(p) => (coss[p as usize], sins[p as usize]),
                        Angle::Fixed { cos, sin } => (cos, sin),
                    };
                    // Every entry's number coeff·sⁿ·cᵐ, once per step, for every
                    // kind of block (there are only a handful).
                    let factors: Vec<Vec<f64>> = self
                        .block_progs
                        .iter()
                        .map(|p| {
                            p.entries
                                .iter()
                                .map(|&(_, _, k, ns, nc)| k * s.powi(ns as i32) * c.powi(nc as i32))
                                .collect()
                        })
                        .collect();
                    for blk in &self.blocks[first as usize..last as usize] {
                        let prog = &self.block_progs[blk.prog as usize];
                        let size = prog.size as usize;
                        let mut old = [0.0f64; 4];
                        for m in 0..size {
                            if blk.members[m] != NONE {
                                old[m] = v[blk.members[m] as usize];
                            }
                        }
                        let mut new = [0.0f64; 4];
                        for (&(out, inp, ..), f) in prog.entries.iter().zip(&factors[blk.prog as usize]) {
                            new[out as usize] += f * old[inp as usize];
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
        self.readout.iter().map(|&b| v[b as usize]).sum()
    }

    /// How many parameters an evaluation must be given: one more than the
    /// largest index a step uses.
    pub(crate) fn n_params(&self) -> usize {
        self.steps
            .iter()
            .filter_map(|s| match *s {
                Step::Rotate { angle: Angle::Param(p), .. } | Step::Block { angle: Angle::Param(p), .. } => {
                    Some(p as usize + 1)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{Block, BlockProg};
    use super::*;

    fn tape(n_boxes: u32, init: Vec<(u32, f64)>, steps: Vec<Step>, pairs: Vec<[u32; 2]>,
            blocks: Vec<Block>, block_progs: Vec<BlockProg>, readout: Vec<u32>) -> Tape {
        Tape { n_boxes, init, steps, pairs, negs: Vec::new(), blocks, block_progs, readout, k2: -1, sine_order: false }
    }

    /// A pair rotation written as a 2-member block: new a = c·a − s·b, new b = s·a + c·b.
    fn pair_prog() -> BlockProg {
        BlockProg { size: 2, entries: vec![(0, 0, 1.0, 0, 1), (0, 1, -1.0, 1, 0), (1, 0, 1.0, 1, 0), (1, 1, 1.0, 0, 1)] }
    }

    #[test]
    fn block_of_two_is_a_rotation() {
        let (s, c) = 0.7f64.sin_cos();
        let init = vec![(0, 0.3), (1, -1.1)];
        let rotate = tape(2, init.clone(), vec![Step::Rotate { angle: Angle::Param(0), first: 0, last: 1 }],
                          vec![[0, 1]], vec![], vec![], vec![0, 1]);
        let block = tape(2, init, vec![Step::Block { angle: Angle::Param(0), first: 0, last: 1 }],
                         vec![], vec![Block { prog: 0, members: [0, 1, NONE, NONE] }], vec![pair_prog()], vec![0, 1]);
        assert!((rotate.value(&[s], &[c]) - block.value(&[s], &[c])).abs() < 1e-15);
    }

    #[test]
    fn none_member_reads_zero_and_is_not_written() {
        // Member 1 has no box: a gets only c·a, and box 1 (another word) is untouched.
        let (s, c) = (-0.4f64).sin_cos();
        let t = tape(2, vec![(0, 2.0), (1, 5.0)], vec![Step::Block { angle: Angle::Param(0), first: 0, last: 1 }],
                     vec![], vec![Block { prog: 0, members: [0, NONE, NONE, NONE] }], vec![pair_prog()], vec![0]);
        assert!((t.value(&[s], &[c]) - c * 2.0).abs() < 1e-15);
        let t = tape(2, vec![(0, 2.0), (1, 5.0)], vec![Step::Block { angle: Angle::Param(0), first: 0, last: 1 }],
                     vec![], vec![Block { prog: 0, members: [0, NONE, NONE, NONE] }], vec![pair_prog()], vec![1]);
        assert_eq!(t.value(&[s], &[c]), 5.0);
    }

    #[test]
    fn block_of_four_with_squares() {
        // CRX's group (IZ, IY, ZZ, ZY) as members 0..4, entries straight from crx_rule
        // (word -> coeff·sⁿ·cᵐ·out, i.e. new[out] += ... old[word]).
        let label = |c: u8, t: u8| match (c, t) { (0, 2) => 0, (0, 3) => 1, (2, 2) => 2, (2, 3) => 3, _ => unreachable!() };
        let mut entries = Vec::new();
        for (word, code) in [(0u8, 2u8), (1, 3), (2, 10), (3, 11)] {
            for &(oc, ot, k, ns, nc) in crate::gates::crx_rule(code).unwrap() {
                entries.push((label(oc, ot), word, k, ns, nc));
            }
        }
        let prog = BlockProg { size: 4, entries };
        let x = [0.3, -0.8, 1.2, 0.5];
        let (s, c) = 1.1f64.sin_cos();
        let t = tape(4, (0..4).map(|i| (i, x[i as usize])).collect(),
                     vec![Step::Block { angle: Angle::Param(0), first: 0, last: 1 }],
                     vec![], vec![Block { prog: 0, members: [0, 1, 2, 3] }], vec![prog], vec![0]);
        // new IZ: from IZ c², from IY −sc, from ZZ s², from ZY +sc (read off crx_rule's columns)
        let expected = c * c * x[0] - s * c * x[1] + s * s * x[2] + s * c * x[3];
        assert!((t.value(&[s], &[c]) - expected).abs() < 1e-15);
    }
}
