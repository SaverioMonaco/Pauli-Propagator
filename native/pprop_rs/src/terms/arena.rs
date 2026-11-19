//! How the sum-of-terms engine stores a term: a coefficient times a chain of
//! sin/cos factors, kept in per-thread arenas of nodes.

use crate::threads::run_all;
use crate::WordMap;
use std::sync::atomic::{AtomicU64, Ordering};

// A term during propagation: a coefficient times the monomial `mono` in a
// `Monomials` arena, which has `depth` sin/cos factors - or only sin factors
// when the arena counts sine order (kept here rather than in the arena so
// the k2 cutoff never has to look the monomial up). Plain
// `Copy` data, 16 bytes, so branching a Pauli word copies each term instead
// of cloning two heap-allocated index lists. A Pauli word's full coefficient
// is a Vec<Term> - a sum of these - since different gates along the circuit
// can each contribute their own factor.
#[derive(Clone, Copy)]
pub(crate) struct Term {
    pub(crate) coeff: f64,
    pub(crate) mono: MonoRef,
    pub(crate) depth: u32,
}

impl Term {
    #[inline]
    pub(crate) fn scaled(self, factor: f64) -> Term {
        Term {
            coeff: factor * self.coeff,
            ..self
        }
    }

    /// This term times one more sin/cos factor of `param`.
    #[inline]
    pub(crate) fn times(self, mono: &mut Monomials, param: u32, is_sin: bool) -> Term {
        Term {
            coeff: self.coeff,
            mono: mono.push(self.mono, param, is_sin),
            depth: self.depth + (is_sin || !mono.sine_order) as u32,
        }
    }
}

// ---------------------------------------------------------------------
// Monomial arenas
// ---------------------------------------------------------------------
//
// Every monomial is a node holding one sin/cos factor and a link to the
// monomial it extends; `ROOT` is the empty product. Appending a factor is
// one push, and the sin and cos branches of a rotation share their parent's
// whole chain instead of each copying it. A node's factors are read back by
// walking to the root, which yields them newest first; `to_lists` reverses
// that, so the index lists come out in the same order they were appended in.
//
// Each propagation thread appends to an arena of its own, so a node
// reference (`MonoRef`) carries the arena it lives in in its top
// `ARENA_BITS` bits. A chain can cross arenas: a term built on one thread
// can be extended on another after it has moved to that thread's shard.
//
// Nodes are never freed individually. Terms dropped by pruning or
// truncation leave dead nodes behind, so `compact` rebuilds the arenas
// from the live terms once they have grown enough (see `heisenberg_one`).

pub(crate) type MonoRef = u32;

pub(crate) const ARENA_BITS: u32 = 5;
pub(crate) const INDEX_BITS: u32 = 32 - ARENA_BITS;
/// Most threads one propagation will use (one arena each).
pub(crate) const MAX_ARENAS: usize = 1 << ARENA_BITS;
/// Most nodes one arena can hold (8 bytes each, so 1 GiB).
pub(crate) const ARENA_CAPACITY: usize = 1 << INDEX_BITS;

#[inline]
pub(crate) fn mono_ref(arena: usize, index: usize) -> MonoRef {
    ((arena as u32) << INDEX_BITS) | index as u32
}

#[inline]
pub(crate) fn split_ref(r: MonoRef) -> (usize, usize) {
    (
        (r >> INDEX_BITS) as usize,
        (r & (ARENA_CAPACITY as u32 - 1)) as usize,
    )
}

#[derive(Clone, Copy)]
pub(crate) struct Node {
    parent: MonoRef,
    factor: u32, // param << 1 | is_sin
}

/// Node 0 of arena 0.
pub(crate) const ROOT: MonoRef = 0;
// Compaction never runs below this many nodes in total.
pub(crate) const MIN_COMPACT_NODES: usize = 1 << 23;

pub(crate) struct Monomials {
    arena: usize,
    nodes: Vec<Node>,
    // Whether `Term::depth` counts only sin factors (k2 as a sine order).
    sine_order: bool,
}

impl Monomials {
    pub(crate) fn new(arena: usize, sine_order: bool) -> Self {
        let mut nodes = Vec::new();
        if arena == 0 {
            nodes.push(Node {
                parent: ROOT,
                factor: 0,
            });
        }
        Monomials {
            arena,
            nodes,
            sine_order,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub(crate) fn push(&mut self, parent: MonoRef, param: u32, is_sin: bool) -> MonoRef {
        let index = self.nodes.len();
        assert!(
            index < ARENA_CAPACITY,
            "pprop_rs: a monomial arena outgrew {} nodes within one layer",
            ARENA_CAPACITY
        );
        self.nodes.push(Node {
            parent,
            factor: (param << 1) | is_sin as u32,
        });
        mono_ref(self.arena, index)
    }

    pub(crate) fn term_from_lists(&mut self, coeff: f64, sin_idx: &[u32], cos_idx: &[u32]) -> Term {
        let mut term = Term {
            coeff,
            mono: ROOT,
            depth: 0,
        };
        for &p in sin_idx {
            term = term.times(self, p, true);
        }
        for &p in cos_idx {
            term = term.times(self, p, false);
        }
        term
    }
}

#[inline]
pub(crate) fn node_at(arenas: &[Monomials], r: MonoRef) -> Node {
    let (a, i) = split_ref(r);
    arenas[a].nodes[i]
}

pub(crate) fn to_lists(arenas: &[Monomials], term: &Term) -> (Vec<u32>, Vec<u32>) {
    let depth = term.depth as usize;
    let mut sin_idx = Vec::with_capacity(depth);
    let mut cos_idx = Vec::with_capacity(depth);
    let mut id = term.mono;
    while id != ROOT {
        let node = node_at(arenas, id);
        if node.factor & 1 == 1 {
            sin_idx.push(node.factor >> 1);
        } else {
            cos_idx.push(node.factor >> 1);
        }
        id = node.parent;
    }
    sin_idx.reverse();
    cos_idx.reverse();
    (sin_idx, cos_idx)
}

/// Rebuild every arena from the nodes the terms in `shards` still reach,
/// and rewrite each term's reference to match. Nodes keep their arena and
/// their order, so a live node's new index is the number of live nodes
/// before it in its arena (its rank), and shared prefixes stay shared.
/// Each step runs one thread per shard or arena when `parallel`.
pub(crate) fn compact<const NW: usize>(arenas: &mut [Monomials], shards: &mut [WordMap<NW>], parallel: bool) {
    // 1. Mark every node some term reaches. A walk stops at the first node
    // that was already marked; whoever marked it walks the rest of its chain.
    let marks: Vec<Vec<AtomicU64>> = arenas
        .iter()
        .map(|a| {
            (0..a.len().div_ceil(64))
                .map(|_| AtomicU64::new(0))
                .collect()
        })
        .collect();
    let mark = |r: MonoRef| -> bool {
        let (a, i) = split_ref(r);
        let bit = 1u64 << (i % 64);
        marks[a][i / 64].fetch_or(bit, Ordering::Relaxed) & bit != 0
    };
    mark(ROOT);
    let old: &[Monomials] = arenas;
    run_all(shards.iter().collect(), parallel, |shard: &WordMap<NW>| {
        for term in shard.values().flatten() {
            let mut id = term.mono;
            while !mark(id) {
                id = node_at(old, id).parent;
            }
        }
    });

    // 2. Rank: the number of live nodes before each 64-node block.
    let live: Vec<Vec<u64>> = marks
        .into_iter()
        .map(|bits| bits.into_iter().map(AtomicU64::into_inner).collect())
        .collect();
    let block_rank: Vec<Vec<u32>> = live
        .iter()
        .map(|bits| {
            let mut below = 0u32;
            bits.iter()
                .map(|b| {
                    let rank = below;
                    below += b.count_ones();
                    rank
                })
                .collect()
        })
        .collect();
    let renumber = |r: MonoRef| -> MonoRef {
        let (a, i) = split_ref(r);
        let before = live[a][i / 64] & ((1u64 << (i % 64)) - 1);
        mono_ref(a, (block_rank[a][i / 64] + before.count_ones()) as usize)
    };

    // 3. Copy each arena's live nodes, in order, with renumbered parents.
    let kept: Vec<Vec<Node>> = run_all(
        old.iter().zip(&live).collect(),
        parallel,
        |(arena, bits): (&Monomials, &Vec<u64>)| {
            let n_live = bits.iter().map(|b| b.count_ones() as usize).sum();
            let mut nodes = Vec::with_capacity(n_live);
            for (w, &word) in bits.iter().enumerate() {
                let mut b = word;
                while b != 0 {
                    let node = arena.nodes[w * 64 + b.trailing_zeros() as usize];
                    nodes.push(Node {
                        parent: renumber(node.parent),
                        ..node
                    });
                    b &= b - 1;
                }
            }
            nodes
        },
    );

    // 4. Point every term at its node's new reference.
    run_all(
        shards.iter_mut().collect(),
        parallel,
        |shard: &mut WordMap<NW>| {
            for term in shard.values_mut().flatten() {
                term.mono = renumber(term.mono);
            }
        },
    );
    for (arena, nodes) in arenas.iter_mut().zip(kept) {
        arena.nodes = nodes;
    }
}
