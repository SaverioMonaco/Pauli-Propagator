//! Cleaning up a compiled tape: drop every update whose result never reaches
//! the readout, then renumber the boxes that are left densely.
//!
//! The compiler writes a pair, single, negation or block for every word a gate
//! changes, including words that are later dropped (k1, pruners, k2,
//! threshold) or that never come back to I/Z. Walking the tape backwards from
//! the readout finds what matters: a box is *live* at some point if its value
//! there is read on the way to the readout.
//!
//! ```text
//! live = readout boxes
//! backwards over the steps:
//!   pair (a, b):  kept if a or b is live; then both are live before it
//!                 (each new value reads both old ones)
//!   single b, negation b:  kept if b is live (b stays live)
//!   block:        kept if any present member is live; then all of them are
//! init:           kept for live boxes only
//! ```
//!
//! What is removed only ever wrote values nobody reads, so values and
//! gradients do not change. Renumbering in order of first use then makes the
//! evaluation arrays as small as the boxes actually used.

use super::{Step, Tape, NONE};

impl Tape {
    /// Remove what cannot reach the readout and renumber the boxes. Returns,
    /// for every old box, its new number (or `NONE` if it is gone).
    pub(crate) fn cleanup(&mut self) -> Vec<u32> {
        let n = self.n_boxes as usize;
        let mut live = vec![false; n];
        for &b in &self.readout {
            live[b as usize] = true;
        }
        // keep_*[i]: whether entry i of that array survives.
        let mut keep_pair = vec![false; self.pairs.len()];
        let mut keep_single = vec![false; self.singles.len()];
        let mut keep_neg = vec![false; self.negs.len()];
        let mut keep_block = vec![false; self.blocks.len()];
        for step in self.steps.iter().rev() {
            match *step {
                Step::Rotate { first, last, sfirst, slast, .. } => {
                    for i in (sfirst..slast).rev() {
                        keep_single[i as usize] = live[self.singles[i as usize] as usize];
                    }
                    for i in (first..last).rev() {
                        let [a, b] = self.pairs[i as usize];
                        let (a, b) = (a as usize, b as usize);
                        if live[a] || live[b] {
                            keep_pair[i as usize] = true;
                            live[a] = true;
                            live[b] = true;
                        }
                    }
                }
                Step::Negate { first, last } => {
                    for i in first..last {
                        keep_neg[i as usize] = live[self.negs[i as usize] as usize];
                    }
                }
                Step::Block { first, last, .. } => {
                    for i in (first..last).rev() {
                        let blk = &self.blocks[i as usize];
                        let size = self.block_progs[blk.prog as usize].size as usize;
                        let present = || blk.members[..size].iter().filter(|&&m| m != NONE);
                        if present().any(|&m| live[m as usize]) {
                            keep_block[i as usize] = true;
                            present().for_each(|&m| live[m as usize] = true);
                        }
                    }
                }
            }
        }
        self.init.retain(|&(b, _)| live[b as usize]);

        // New box numbers, in order of first use going forward.
        let mut renumber = vec![NONE; n];
        let mut next = 0u32;
        let mut number = |b: u32| -> u32 {
            if renumber[b as usize] == NONE {
                renumber[b as usize] = next;
                next += 1;
            }
            renumber[b as usize]
        };
        for &(b, _) in &self.init {
            number(b);
        }

        // Rebuild every array and step with the kept entries, renumbered.
        let (mut pairs, mut singles, mut negs, mut blocks) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut steps = Vec::with_capacity(self.steps.len());
        for step in &self.steps {
            match *step {
                Step::Rotate { ref angle, first, last, sfirst, slast } => {
                    let (f, s) = (pairs.len() as u32, singles.len() as u32);
                    for i in first..last {
                        if keep_pair[i as usize] {
                            let [a, b] = self.pairs[i as usize];
                            pairs.push([number(a), number(b)]);
                        }
                    }
                    for i in sfirst..slast {
                        if keep_single[i as usize] {
                            singles.push(number(self.singles[i as usize]));
                        }
                    }
                    let (l, sl) = (pairs.len() as u32, singles.len() as u32);
                    if l > f || sl > s {
                        steps.push(Step::Rotate { angle: angle.clone(), first: f, last: l, sfirst: s, slast: sl });
                    }
                }
                Step::Negate { first, last } => {
                    let f = negs.len() as u32;
                    for i in first..last {
                        if keep_neg[i as usize] {
                            negs.push(number(self.negs[i as usize]));
                        }
                    }
                    let l = negs.len() as u32;
                    if l > f {
                        steps.push(Step::Negate { first: f, last: l });
                    }
                }
                Step::Block { ref angle, first, last } => {
                    let f = blocks.len() as u32;
                    for i in first..last {
                        if keep_block[i as usize] {
                            let mut blk = self.blocks[i as usize].clone();
                            for m in blk.members.iter_mut() {
                                if *m != NONE {
                                    *m = number(*m);
                                }
                            }
                            blocks.push(blk);
                        }
                    }
                    let l = blocks.len() as u32;
                    if l > f {
                        steps.push(Step::Block { angle: angle.clone(), first: f, last: l });
                    }
                }
            }
        }
        for b in self.readout.iter_mut() {
            *b = number(*b);
        }
        for (b, _) in self.init.iter_mut() {
            *b = renumber[*b as usize];
        }
        self.steps = steps;
        self.pairs = pairs;
        self.singles = singles;
        self.negs = negs;
        self.blocks = blocks;
        self.n_boxes = next;
        renumber
    }
}
