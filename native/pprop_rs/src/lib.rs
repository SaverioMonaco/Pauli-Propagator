//! Rust implementation of pprop's Heisenberg propagation core.
//!
//! This is the *only* propagation backend in this fork of pprop (see the
//! project README / paper appendix for why the pure-Python
//! `heisenberg()`/`PauliDict` implementation was removed rather than kept
//! alongside this one). It covers every gate in `pprop.gates`
//! (RX/RY/RZ, H/S/SX/T, SWAP, CNOT/CY/CZ, CRX/CRY/CRZ), both exact pruners
//! (`DeadQubitPruner`, `XYWeightPruner`), and all three truncations
//! (`WeightTruncation`, `FrequencyTruncation`, `CoefficientTruncation`).
//! The evolution rule tables below are transcribed 1:1 from the `rule` dicts
//! in `pprop/gates/*.py`, which remain in the Python source as the
//! human-readable reference these tables are checked against, entry-for-entry,
//! by `tests/test_rule_tables.py` (via `evolve_single_gate_debug` below).
//! `tests/test_backends.py` covers correctness at the full-circuit level
//! instead (random circuits vs. PennyLane directly).
//!
//! Pauli-word representation: each x/z plane is a fixed-size `[u64; NW]`
//! array (one bit per qubit, `NW` 64-bit words), where `NW` is chosen per
//! circuit as the smallest power of two covering `num_qubits`
//! (`words_needed`). `NW` is a *const generic* parameter, so every function
//! below is monomorphized once per supported `NW` and the array stays
//! stack-allocated (no heap allocation per Pauli word, same as the old
//! single-`u64` version) - only the word count grows with the circuit.
//! `propagate_batch`/`evolve_single_gate_debug` dispatch to the right
//! monomorphization at runtime (`dispatch_nw!`). The supported sizes top out
//! at `NW = 128` (8192 qubits); add a literal to `dispatch_nw!` to raise
//! that ceiling. The pure Python `PauliOp` used arbitrary-precision ints and
//! had no limit at all - this is a wide, but still finite, stand-in for
//! that.
//!
//! Concretely, for qubit index `wire`, its bit lives in word `wire / 64` at
//! bit position `wire % 64` (see `word_bit` below) - word 0 covers qubits
//! 0-63, word 1 covers 64-127, and so on. Worked example, `NW = 2` (covers
//! qubits 0-127), Pauli word "Y on qubit 0, Z on qubit 5, X on qubit 70,
//! I elsewhere":
//!
//! ```text
//! qubit 0 (Y: x-bit=1, z-bit=1) -> word 0/64=0, bit 0
//! qubit 5 (Z: x-bit=0, z-bit=1) -> word 5/64=0, bit 5
//! qubit 70 (X: x-bit=1, z-bit=0) -> word 70/64=1, bit 70%64=6
//!
//! x = [0b0000001, 0b1000000] = [1, 64]   // word0 bit0 (qubit 0's x-part); word1 bit6 (qubit 70)
//! z = [0b0100001, 0b0000000] = [33, 0]   // word0 bit0+bit5 (qubit 0's z-part, qubit 5); word1 = 0
//! ```
//!
//! That one `PauliKey<2>` value, `([1, 64], [33, 0])`, *is* "Y0 Z5 X70" -
//! nothing else is stored anywhere for it. Reading qubit 70 back out: word 1
//! bit 6 -> `(x[1] >> 6) & 1 = 1`, `(z[1] >> 6) & 1 = 0` -> x-bit set,
//! z-bit clear -> X.

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyByteArray;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

// One trigonometric product term as Python sees it: (coeff, sin_indices,
// cos_indices) means coeff * prod(sin(theta_i) for i in sin_indices) *
// prod(cos(theta_j) for j in cos_indices), e.g. (0.5, [0], [1]) =
// 0.5*sin(theta_0)*cos(theta_1), and (-0.25, [], [0, 0]) =
// -0.25*cos(theta_0)^2 (repeated index = squared). This is only the
// boundary format; inside the propagation loop a term is a `Term`.
type CoeffTerm = (f64, Vec<u32>, Vec<u32>);
// A term during propagation: a coefficient times the monomial `mono` in a
// `Monomials` arena, which has `depth` sin/cos factors (kept here rather than
// in the arena so the k2 cutoff never has to look the monomial up). Plain
// `Copy` data, 16 bytes, so branching a Pauli word copies each term instead
// of cloning two heap-allocated index lists. A Pauli word's full coefficient
// is a Vec<Term> - a sum of these - since different gates along the circuit
// can each contribute their own factor.
#[derive(Clone, Copy)]
struct Term {
    coeff: f64,
    mono: MonoRef,
    depth: u32,
}

impl Term {
    #[inline]
    fn scaled(self, factor: f64) -> Term {
        Term {
            coeff: factor * self.coeff,
            ..self
        }
    }

    /// This term times one more sin/cos factor of `param`.
    #[inline]
    fn times(self, mono: &mut Monomials, param: u32, is_sin: bool) -> Term {
        Term {
            coeff: self.coeff,
            mono: mono.push(self.mono, param, is_sin),
            depth: self.depth + 1,
        }
    }
}
// (x, z) bitmask planes, same convention as pprop.pauli.op.PauliOp, packed
// into NW little-endian u64 words instead of one u64 (see module docs above
// for the worked "Y0 Z5 X70" example). A PauliKey<NW> value fully identifies
// one Pauli word; FxHashMap<PauliKey<NW>, Vec<Term>> is "one Pauli word
// -> its current coefficient", the data structure the whole propagation
// loop below updates at every gate step.
type PauliKey<const NW: usize> = ([u64; NW], [u64; NW]);
// Output of evolving words through one gate, before truncation and insertion.
type Emitted<const NW: usize> = Vec<(PauliKey<NW>, Vec<Term>)>;
// Pauli word -> its current coefficient, for the words one thread owns.
type WordMap<const NW: usize> = FxHashMap<PauliKey<NW>, Vec<Term>>;

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

type MonoRef = u32;

const ARENA_BITS: u32 = 5;
const INDEX_BITS: u32 = 32 - ARENA_BITS;
/// Most threads one propagation will use (one arena each).
const MAX_ARENAS: usize = 1 << ARENA_BITS;
/// Most nodes one arena can hold (8 bytes each, so 1 GiB).
const ARENA_CAPACITY: usize = 1 << INDEX_BITS;

#[inline]
fn mono_ref(arena: usize, index: usize) -> MonoRef {
    ((arena as u32) << INDEX_BITS) | index as u32
}

#[inline]
fn split_ref(r: MonoRef) -> (usize, usize) {
    (
        (r >> INDEX_BITS) as usize,
        (r & (ARENA_CAPACITY as u32 - 1)) as usize,
    )
}

#[derive(Clone, Copy)]
struct Node {
    parent: MonoRef,
    factor: u32, // param << 1 | is_sin
}

/// Node 0 of arena 0.
const ROOT: MonoRef = 0;
// Compaction never runs below this many nodes in total.
const MIN_COMPACT_NODES: usize = 1 << 23;

struct Monomials {
    arena: usize,
    nodes: Vec<Node>,
}

impl Monomials {
    fn new(arena: usize) -> Self {
        let mut nodes = Vec::new();
        if arena == 0 {
            nodes.push(Node {
                parent: ROOT,
                factor: 0,
            });
        }
        Monomials { arena, nodes }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    fn push(&mut self, parent: MonoRef, param: u32, is_sin: bool) -> MonoRef {
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

    fn term_from_lists(&mut self, coeff: f64, sin_idx: &[u32], cos_idx: &[u32]) -> Term {
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
fn node_at(arenas: &[Monomials], r: MonoRef) -> Node {
    let (a, i) = split_ref(r);
    arenas[a].nodes[i]
}

fn to_lists(arenas: &[Monomials], term: &Term) -> (Vec<u32>, Vec<u32>) {
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
fn compact<const NW: usize>(arenas: &mut [Monomials], shards: &mut [WordMap<NW>], parallel: bool) {
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

// Gate kind codes, matching pprop.propagator.GATE_KIND in __init__.py.
const RX: u8 = 0;
const RY: u8 = 1;
const RZ: u8 = 2;
const H: u8 = 3;
const S: u8 = 4;
const SX: u8 = 5;
const T: u8 = 6;
const SWAP: u8 = 7;
const CNOT: u8 = 8;
const CY: u8 = 9;
const CZ: u8 = 10;
const CRX: u8 = 11;
const CRY: u8 = 12;
const CRZ: u8 = 13;

struct GateSpec<const NW: usize> {
    kind: u8,
    wire0: u32,
    wire1: i64, // -1 unless two-qubit
    param: i64, // -1 unless read from a trainable parameter slot
    // Constant angle of a rotation/controlled-rotation gate that was given a
    // plain float rather than a trainable index (e.g. `qml.RY(np.pi, wires=0)`).
    // `None` for trainable and non-parametrised gates. See `fixed_sin_cos`.
    fixed: Option<f64>,
    wire_mask: [u64; NW],
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

const FOLD_SNAP: f64 = 1e-12;

#[inline]
fn snap(v: f64) -> f64 {
    if v.abs() < FOLD_SNAP { 0.0 } else { v }
}

/// `(sin(theta), cos(theta))` of a fixed angle, with exact zeros at the
/// quarter turns. Callers pass the angle in whatever convention the gate's
/// rule table is written in (the full angle for RX/RY/RZ, the half angle
/// for CRX/CRY/CRZ - see `controlledrotation.py`).
fn fixed_sin_cos(theta: f64) -> (f64, f64) {
    let (s, c) = theta.sin_cos();
    (snap(s), snap(c))
}

/// Multiply every term's coefficient by a constant, in place.
fn scale_terms(mut terms: Vec<Term>, factor: f64) -> Vec<Term> {
    if factor != 1.0 {
        for t in terms.iter_mut() {
            *t = t.scaled(factor);
        }
    }
    terms
}

// ---------------------------------------------------------------------
// NW-word bit-array helpers (the [u64; NW] equivalents of the old u64 ops)
// ---------------------------------------------------------------------
//
// Bitwise-operator refresher, since almost everything below is one of
// these five, composed. All examples use an 8-bit number for readability;
// the real code works on u64, same idea, just 64 bits instead of 8.
//
//   <<  (left shift)   1u64 << 5  = 0b00100000 = 32
//                       "a single 1 bit, moved to position 5"
//   >>  (right shift)  0b00100000 >> 5 = 0b00000001 = 1
//                       "move the bit at position 5 down to position 0,
//                        so it can be read as a plain 0-or-1 value"
//   &   (AND)          0b00100101 & 0b00100000 = 0b00100000
//                       "keep only the bits that are 1 in BOTH operands -
//                        used to test/extract one bit: value & (1<<b)"
//   |   (OR)           0b00000101 | 0b00100000 = 0b00100101
//                       "combine bits from both operands - used to SET a
//                        bit without disturbing any other bit"
//   !   (NOT)          !0b00100000 = 0b11011111  (all other bits, 8-bit example)
//                       "flip every bit - !(1<<b) is 'every bit except b',
//                        so x &= !(1<<b) clears just bit b and nothing else"
//
// `1u64 << b` on its own, note, is always "a mask with exactly one bit
// set, at position b" - that single pattern (build a one-bit mask, then
// combine it with & to read, or with | / &! to write) is almost the whole
// bit-manipulation vocabulary this file uses.

/// Smallest power-of-two word count whose 64-bit words cover `num_qubits`.
fn words_needed(num_qubits: u32) -> usize {
    let raw = ((num_qubits as usize) + 63) / 64;
    raw.max(1).next_power_of_two()
}

#[inline]
fn word_bit(wire: u32) -> (usize, u32) {
    ((wire / 64) as usize, wire % 64)
}

// `words: &[u64]` is a *slice* (brackets with no length) - a borrowed view
// over a Vec/list of a length not known at compile time, unlike `[u64; NW]`
// whose length IS part of the type. This is the shape Pauli-word data
// arrives in from Python (a plain list, PyO3-converted to Vec<u64>, then
// borrowed here); this function copies it into a proper `[u64; NW]`,
// zero-padding if Python sent fewer words than NW needs (e.g. a length-1
// word list going into an NW=2 array just leaves word 1 = 0, correctly
// meaning "no qubits above 63 are set").
fn words_to_array<const NW: usize>(words: &[u64]) -> [u64; NW] {
    let mut arr = [0u64; NW];
    let n = words.len().min(NW);
    arr[..n].copy_from_slice(&words[..n]);
    arr
}

// Pauli label encoding, identical to PauliOp: bit0 = x-bit, bit1 = z-bit.
// I=0, X=1, Z=2, Y=3. Reads one qubit's label back out of the (x, z)
// arrays: find its (word, bit) with word_bit, pull the x-bit and z-bit out
// of that word, pack them into one u8. Worked example (see the module doc
// comment's "Y0 Z5 X70" Pauli word, x=[1,64], z=[33,0]): pauli_label(x, z,
// 70) -> word_bit(70)=(1,6) -> (x[1]>>6)&1=1, (z[1]>>6)&1=0 -> label=1 (X).
fn pauli_label<const NW: usize>(x: &[u64; NW], z: &[u64; NW], wire: u32) -> u8 {
    let (w, b) = word_bit(wire);

    // Read bit b of x[w]: shift word w right by b places, so the bit we
    // want lands in position 0 (example, qubit 70's X-bit: x[1] = 64 =
    // 0b1000000, b = 6, so x[1] >> 6 = 0b1 = 1). `& 1` then keeps only
    // that bottom bit, discarding anything that was shifted in above it -
    // this "shift the target bit to position 0, then & 1" pair is the
    // standard "read bit b" idiom used everywhere in this file.
    let x_bit = ((x[w] >> b) & 1) as u8;
    // Same "read bit b" idiom for the z-plane word (qubit 70: z[1] = 0,
    // so z_bit = 0 here).
    let z_bit = ((z[w] >> b) & 1) as u8;

    // Pack the two single bits into one u8 label: x_bit occupies bit
    // position 0 as-is, z_bit gets shifted up into bit position 1 first
    // (`z_bit << 1` turns a value of 0 or 1 into 0 or 2) so that `|`
    // combines them without the two overlapping - they now sit in
    // different bit positions of the result. Qubit 70 example: x_bit=1,
    // z_bit=0 -> 1 | (0 << 1) = 1 -> label 1 = X.
    x_bit | (z_bit << 1)
}

// Writes one qubit's label into (x, z): clear both bits at that position,
// then set them from `label`'s two bits. Takes x/z *by value* (mut x/z -
// arrays of Copy types like u64 copy cheaply, no heap involved) and
// returns a new (x, z) pair rather than mutating in place, because callers
// typically need both the unmodified original (for one branch) and this
// modified version (for another) at the same time - see evolve_rotation
// below for exactly that.
//
// Worked example threaded through every line: start from x=[0], z=[0]
// (all-identity, NW=1), call set_label(x, z, wire=5, label=3) to place a
// Y on qubit 5 (label 3 = 0b11 = x-bit 1, z-bit 1, per the I=0/X=1/Z=2/Y=3
// encoding).
fn set_label<const NW: usize>(
    mut x: [u64; NW],
    mut z: [u64; NW],
    wire: u32,
    label: u8,
) -> ([u64; NW], [u64; NW]) {
    let (w, b) = word_bit(wire);   // wire=5 -> w=0, b=5

    // `1u64 << b` is a mask with exactly one bit set, at position b:
    // b=5 -> 0b00100000 = 32. `!(...)` (bitwise NOT) flips every bit, so
    // `!(1u64 << b)` is "every bit set to 1 EXCEPT bit b". AND-ing x[w]
    // with that mask leaves every other bit of x[w] untouched but forces
    // bit b to 0, regardless of what it was - "clear bit b, nothing else".
    // Example: x[0]=0 & !32 = 0 (already 0, no visible change here, but
    // if some other qubit's bit had been set it would survive untouched).
    x[w] &= !(1u64 << b);
    // Same "clear bit b" operation on the z-plane word.
    z[w] &= !(1u64 << b);

    // `label & 1` reads label's own bit 0 (the x-bit of the label to
    // write): label=3=0b11, so label & 1 = 1. `as u64` casts it from u8
    // to u64 so it can be shifted/combined with x[w] (Rust never mixes
    // integer types implicitly). `<< b` moves that single bit up to
    // position b=5: 1 << 5 = 32. OR-ing it into x[w] (already bit-5-clear
    // from the line above) sets bit 5 to this new value without touching
    // any other bit - OR-ing 0 into every other position changes nothing
    // there. Result: x[0] = 0 | 32 = 32.
    x[w] |= ((label & 1) as u64) << b;
    // Same idea for the z-plane, reading label's bit 1 instead: `label >>
    // 1` shifts bit 1 down to position 0 (label=3=0b11 -> 0b1=1), `& 1`
    // isolates it (already isolated here, but this is the general "read
    // bit 1" pattern), giving the z-bit of the label to write (1, for Y).
    // Shifted to position b=5 and OR'd in: z[0] = 0 | 32 = 32.
    z[w] |= (((label >> 1) & 1) as u64) << b;

    // Result: x=[32], z=[32]. Sanity check against pauli_label: word_bit(5)
    // = (0, 5); (x[0]>>5)&1 = 1, (z[0]>>5)&1 = 1 -> label = 1 | (1<<1) = 3
    // = Y. Matches what we set out to write.
    (x, z)
}

// Simpler cousin of set_label: sets one bit, no label decoding involved -
// used for marking "this gate touches this wire" in a gate's wire_mask,
// not for encoding a Pauli operator. `&mut [u64; NW]` is a *mutable
// borrow*, so this modifies the caller's array in place instead of
// returning a new one (no (x, z) pair to juggle here, just one array).
fn set_wire_bit<const NW: usize>(arr: &mut [u64; NW], wire: u32) {
    let (w, b) = word_bit(wire);
    // `1u64 << b` = a mask with only bit b set; OR-ing it into arr[w]
    // forces that one bit to 1 while leaving every other bit exactly as
    // it was (OR with 0 changes nothing). No clearing step needed here,
    // unlike set_label - a wire_mask only ever gets bits added to it,
    // never removed, so there's nothing to clear first.
    arr[w] |= 1u64 << b;
}

// ---------------------------------------------------------------------
// Whole-array (all NW words at once) helpers. New syntax vs. the functions
// above: with a single u64 you could just write `a & b` or `a | b` and be
// done; with `[u64; NW]` there's no built-in "AND/OR every word together"
// operator, so these loop over the NW words by hand, word by word. Two
// ways that loop gets written below - pick whichever reads more naturally
// per function:
//
//   for i in 0..NW { ... a[i] ... }
//       A plain indexed loop. `0..NW` is a *range* (like Python's
//       `range(NW)`), producing 0, 1, ..., NW-1.
//
//   a.iter().zip(b.iter()).any(|(x, y)| ...)
//       The iterator-chain style. `.iter()` walks an array yielding
//       references to each element (Python analogue: `iter(a)`).
//       `.zip(b.iter())` pairs them up with b's elements one-for-one
//       (Python's `zip(a, b)`). `.any(|(x, y)| EXPR)` (Python's
//       `any(EXPR for x, y in zip(a, b))`) returns true as soon as EXPR is
//       true for some pair, short-circuiting - `.all(...)` is the same but
//       requires EXPR true for every pair (Python's `all(...)`), and
//       `.map(f).sum()` transforms each element with `f` then adds them up
//       (Python's `sum(f(x) for x in a)`). `|(x, y)| EXPR` is a *closure*
//       - an inline anonymous function, Rust's equivalent of `lambda x, y:
//       EXPR` - `|params| body` instead of Python's `lambda params: body`.
// ---------------------------------------------------------------------

// Does any word of `a` share a set bit with the corresponding word of `b`?
// Used for the gate-skip fast path: "does this gate's wire_mask overlap
// the current active_mask at all? If not, every Pauli word is identity on
// every wire this gate touches, so the gate can't change anything - skip
// it entirely." Example (NW=1): a=[0b0101], b=[0b0010] -> 0b0101 & 0b0010
// = 0b0000 -> no shared bit -> false. a=[0b0101], b=[0b0001] -> 0b0101 &
// 0b0001 = 0b0001 != 0 -> true.
fn mask_intersects<const NW: usize>(a: &[u64; NW], b: &[u64; NW]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| x & y != 0)
}

// In-place "OR every word of b into a" (mutates a, nothing returned) -
// `&mut` like set_wire_bit earlier, same reasoning. Used to accumulate a
// running union of many separate masks into one existing array without
// allocating a new one each time.
fn mask_or_assign<const NW: usize>(a: &mut [u64; NW], b: &[u64; NW]) {
    for i in 0..NW {
        a[i] |= b[i];
    }
}

// Same "OR every word together" idea as mask_or_assign, but builds and
// returns a brand new array instead of mutating an existing one - used
// where the caller needs to keep `a` unchanged (e.g. building a table of
// several different union results side by side, each one immutable once
// computed).
fn mask_or<const NW: usize>(a: &[u64; NW], b: &[u64; NW]) -> [u64; NW] {
    let mut out = [0u64; NW];
    for i in 0..NW {
        out[i] = a[i] | b[i];
    }
    out
}

// Total number of 1-bits set across every word of `a`. `.count_ones()` is
// a built-in method on integer types - "popcount", a single hardware
// instruction, not a manual bit-by-bit loop. Used for XY-weight (only the
// x-array is passed in: X and Y both set the x-bit, so popcount(x) counts
// exactly the qubits carrying X or Y - same quantity the old `x.count_ones()`
// on a single u64 computed, just summed across NW words now).
fn mask_popcount<const NW: usize>(a: &[u64; NW]) -> u32 {
    a.iter().map(|w| w.count_ones()).sum()
}

// Total Pauli weight: popcount(x | z) summed across words - every qubit
// that's X, Y, *or* Z counts (unlike mask_popcount above, which only
// counted X/Y). `.zip(z.iter())` pairs up x[i] with z[i] word-by-word,
// `(a | b).count_ones()` combines and counts each pair, `.sum()` adds
// those per-word counts into one total.
fn combined_weight<const NW: usize>(x: &[u64; NW], z: &[u64; NW]) -> u32 {
    x.iter().zip(z.iter()).map(|(a, b)| (a | b).count_ones()).sum()
}

// "Is every 1-bit of `a` also a 1-bit of `allowed`?" (a is a subset of
// allowed, as bit sets). Per word: `!al` flips `allowed`'s bits (every bit
// NOT allowed), `x & !al` keeps only the bits of `x` that fall outside
// `allowed`, and `== 0` checks there are none - i.e. no bit of `x` lies
// outside `allowed`. `.all(...)` requires this true for every word, not
// just one. Used by DeadQubitPruner: a Pauli word survives only if every
// qubit where it carries X/Y is still touchable by some remaining gate.
fn mask_subset<const NW: usize>(a: &[u64; NW], allowed: &[u64; NW]) -> bool {
    a.iter().zip(allowed.iter()).all(|(x, al)| x & !al == 0)
}

// Is every word of `a` exactly 0 (the whole array represents "no qubit
// set")? `*w` dereferences `w` (a `&u64` from `.iter()`) back to a plain
// `u64` to compare against `0` - `.iter()` yields references, and `==`
// needs values on the same footing on both sides. This is the NW-word
// generalization of the old single-word `x == 0` check, used by
// `to_expectation` to test whether a Pauli word is entirely Z/I (its
// x-plane is all zero) and therefore contributes to <0|P|0>.
fn mask_is_zero<const NW: usize>(a: &[u64; NW]) -> bool {
    a.iter().all(|w| *w == 0)
}

/// Merge `terms` into `key`'s coefficient. A new word takes ownership of the
/// vector as-is; only a word that already exists pays for a copy.
fn insert_or_extend<const NW: usize>(
    map: &mut FxHashMap<PauliKey<NW>, Vec<Term>>,
    key: PauliKey<NW>,
    mut terms: Vec<Term>,
) {
    use std::collections::hash_map::Entry;
    match map.entry(key) {
        Entry::Vacant(slot) => {
            slot.insert(terms);
        }
        Entry::Occupied(mut slot) => {
            let existing = slot.get_mut();
            if existing.len() < terms.len() {
                std::mem::swap(existing, &mut terms);
            }
            existing.append(&mut terms);
        }
    }
}

// ---------------------------------------------------------------------
// Rule tables (transcribed from pprop/gates/*.py)
// ---------------------------------------------------------------------

// Single-qubit rotations: RX/RY/RZ. rule(label) -> Some((out_label, sign)) for
// the anti-commuting branch, None if `label` commutes with the axis.
fn rx_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        3 => Some((2, -1)), // Y -> Z
        2 => Some((3, 1)),  // Z -> Y
        _ => None,
    }
}
fn ry_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((2, 1)),  // X -> Z
        2 => Some((1, -1)), // Z -> X
        _ => None,
    }
}
fn rz_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((3, -1)), // X -> Y
        3 => Some((1, 1)),  // Y -> X
        _ => None,
    }
}

// Single-qubit Clifford gates: H/S/SX. rule(label) -> Some((out_label, sign)).
// Always exactly one output branch, since these are non-parametrised and
// don't split into a sin/cos pair the way rotations do.
fn h_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((2, 1)),  // X -> Z
        3 => Some((3, -1)), // Y -> -Y
        2 => Some((1, 1)),  // Z -> X
        _ => None,
    }
}
fn s_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        1 => Some((3, -1)), // X -> -Y
        3 => Some((1, 1)),  // Y -> X
        _ => None,
    }
}
fn sx_rule(label: u8) -> Option<(u8, i8)> {
    match label {
        3 => Some((2, -1)), // Y -> -Z
        2 => Some((3, 1)),  // Z -> Y
        _ => None,
    }
}

// T gate: single-qubit non-Clifford, splits into two branches with constant
// (non-trigonometric) phases.
fn t_rule(label: u8) -> Option<[(u8, f64); 2]> {
    const INV_SQRT2: f64 = std::f64::consts::FRAC_1_SQRT_2;
    match label {
        1 => Some([(1, INV_SQRT2), (3, -INV_SQRT2)]),  // X -> +X/sqrt2 - Y/sqrt2
        3 => Some([(3, INV_SQRT2), (1, INV_SQRT2)]),   // Y -> +Y/sqrt2 + X/sqrt2
        _ => None,
    }
}

// Two-qubit non-parametrised controlled gates: CNOT/CY/CZ. rule(code) ->
// Some((out_control, out_target, sign)), code = (control_label << 2) | target_label.
fn cnot_rule(code: u8) -> Option<(u8, u8, i8)> {
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
fn cy_rule(code: u8) -> Option<(u8, u8, i8)> {
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
fn cz_rule(code: u8) -> Option<(u8, u8, i8)> {
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
type CRRule = &'static [(u8, u8, f64, u8, u8)];
fn crx_rule(code: u8) -> Option<CRRule> {
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
fn cry_rule(code: u8) -> Option<CRRule> {
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
fn crz_rule(code: u8) -> Option<CRRule> {
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
fn evolve_rotation<const NW: usize>(
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
fn apply_sign(mut terms: Vec<Term>, sign: i8) -> Vec<Term> {
    if sign != 1 {
        for t in terms.iter_mut() {
            t.coeff = -t.coeff;
        }
    }
    terms
}

/// H/S/SX shape: commuting Pauli passes through; otherwise exactly one
/// output word with a constant +-1 sign (no trig, no branching).
fn evolve_clifford1q<const NW: usize>(
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
fn evolve_t<const NW: usize>(
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

fn evolve_swap<const NW: usize>(
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
fn evolve_controlled<const NW: usize>(
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
fn evolve_controlled_rotation<const NW: usize>(
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

/// Order of the terms in a finished expression: by index lists, then by
/// coefficient.
fn cmp_terms(a: &CoeffTerm, b: &CoeffTerm) -> std::cmp::Ordering {
    (&a.1, &a.2)
        .cmp(&(&b.1, &b.2))
        .then_with(|| a.0.total_cmp(&b.0))
}

/// The terms of every word with no X/Y, i.e. with a nonzero expectation in
/// |0...0>, sorted by `cmp_terms`. Iterating the maps gives an order that
/// depends on hashing and on how the words were split over threads; sorting
/// makes the expression the same either way.
fn to_expectation<const NW: usize>(
    shards: &[WordMap<NW>],
    arenas: &[Monomials],
    parallel: bool,
) -> Vec<CoeffTerm> {
    let runs = run_all(shards.iter().collect(), parallel, |shard: &WordMap<NW>| {
        let mut run = Vec::new();
        for ((x, _z), terms) in shard.iter() {
            if mask_is_zero(x) {
                for t in terms {
                    let (s, cc) = to_lists(arenas, t);
                    run.push((t.coeff, s, cc));
                }
            }
        }
        run.sort_unstable_by(cmp_terms);
        run
    });
    let mut out: Vec<CoeffTerm> = runs.into_iter().flatten().collect();
    // A stable sort finds the per-shard runs and just merges them.
    out.sort_by(cmp_terms);
    out
}

/// Whether `gate` changes the word `(x, z)` at all, i.e. whether its rule
/// table has an entry for the word's labels on the gate's wires. Words it
/// does not act on keep their key and terms unchanged.
fn gate_acts<const NW: usize>(gate: &GateSpec<NW>, x: &[u64; NW], z: &[u64; NW]) -> bool {
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

fn evolve_one_gate<const NW: usize>(
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

// ---------------------------------------------------------------------
// Heisenberg propagation loop
// ---------------------------------------------------------------------

/// XYWeightPruner verdict: can the gates in `gate_masks` (the rest of the
/// circuit, in order) still reduce this word's XY-weight to zero? Each gate
/// that touches the growing causal cone of `x` can remove at most one X/Y.
fn xy_reachable<const NW: usize>(x: &[u64; NW], gate_masks: &[[u64; NW]]) -> bool {
    let xy_weight = mask_popcount(x);
    if xy_weight == 0 {
        return true;
    }
    let mut xy_support = *x;
    let mut budget = 0u32;
    for gm in gate_masks {
        if mask_intersects(gm, &xy_support) {
            budget += 1;
            mask_or_assign(&mut xy_support, gm);
            if budget >= xy_weight {
                return true;
            }
        }
    }
    false
}

/// Apply the k1/k2/coefficient truncations to one word; `false` means the
/// word has nothing left and should be dropped.
fn truncate<const NW: usize>(
    key: &PauliKey<NW>,
    terms: &mut Vec<Term>,
    k1: i64,
    k2: i64,
    coeff_threshold: f64,
) -> bool {
    if k1 >= 0 && combined_weight(&key.0, &key.1) > k1 as u32 {
        return false;
    }
    if k2 >= 0 {
        let k2u = k2 as u32;
        terms.retain(|t| t.depth <= k2u);
    }
    if coeff_threshold >= 0.0 {
        terms.retain(|t| t.coeff.abs() >= coeff_threshold);
    }
    !terms.is_empty()
}

/// A run of consecutive gates (in propagation order) on pairwise disjoint
/// wires. Such gates commute, and a word's labels on one gate's wires are
/// untouched by the others, so the gates a word will meet in the layer can
/// be read off the word once, before any of them is applied.
struct Layer<const NW: usize> {
    start: usize,
    end: usize,
    mask: [u64; NW],
}

/// Split `gate_masks` greedily into maximal runs of wire-disjoint gates.
fn build_layers<const NW: usize>(gate_masks: &[[u64; NW]]) -> Vec<Layer<NW>> {
    let mut layers: Vec<Layer<NW>> = Vec::new();
    for (i, gm) in gate_masks.iter().enumerate() {
        match layers.last_mut() {
            Some(layer) if !mask_intersects(&layer.mask, gm) => {
                layer.end = i + 1;
                mask_or_assign(&mut layer.mask, gm);
            }
            _ => layers.push(Layer {
                start: i,
                end: i + 1,
                mask: *gm,
            }),
        }
    }
    layers
}

/// Everything about the circuit and the truncation settings that stays fixed
/// for a whole propagation, shared read-only by every thread.
struct Circuit<'a, const NW: usize> {
    reversed: &'a [GateSpec<NW>],
    gate_masks: &'a [[u64; NW]],
    active_qubits_from: &'a [[u64; NW]],
    k1: i64,
    k2: i64,
    coeff_threshold: f64, // < 0 disables
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
}

/// One layer's pass, as seen by each shard's thread.
struct LayerPass<'a, const NW: usize> {
    circuit: &'a Circuit<'a, NW>,
    layer: &'a Layer<NW>,
    // Gate owning each qubit in this layer (usize::MAX = none).
    gate_on_qubit: &'a [usize],
    // Whether the map still holds the observable's untruncated words.
    fresh: bool,
    first_gate: Option<usize>,
    n_shards: usize,
}

// Below this many words a layer runs on the calling thread; spawning
// would cost more than it saves.
const PARALLEL_MIN_WORDS: usize = 1 << 13;

/// The shard (and so the thread) that owns `key`. The hash is remixed so
/// the keys within one shard still spread over all of its map's buckets.
fn shard_of<const NW: usize>(key: &PauliKey<NW>, n_shards: usize) -> usize {
    if n_shards == 1 {
        return 0;
    }
    let mut h = FxHasher::default();
    key.hash(&mut h);
    let mixed = h.finish().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
    ((mixed * n_shards as u64) >> 32) as usize
}

/// First half of a layer, for one shard: move out the words the layer acts
/// on, carry each through its gates in circuit order, and batch the results
/// by the shard that owns them. Also returns the support of the words that
/// stayed.
fn evolve_shard<const NW: usize>(
    pass: &LayerPass<NW>,
    shard: &mut WordMap<NW>,
    mono: &mut Monomials,
) -> (Vec<Emitted<NW>>, [u64; NW]) {
    let c = pass.circuit;
    let truncate = |key: &PauliKey<NW>, terms: &mut Vec<Term>| {
        truncate(key, terms, c.k1, c.k2, c.coeff_threshold)
    };

    // Each moved-out word comes with the (ascending) indices of the gates
    // that act on it.
    let mut acted: Vec<(PauliKey<NW>, Vec<Term>, Vec<usize>)> = Vec::new();
    let mut stayed_active = [0u64; NW];
    shard.retain(|key, terms| {
        let (x, z) = key;
        let support = mask_or(x, z);
        let mut gates: Vec<usize> = Vec::new();
        if mask_intersects(&support, &pass.layer.mask) {
            for_each_bit(&support, |q| {
                let g = pass.gate_on_qubit[q];
                if g != usize::MAX && !gates.contains(&g) && gate_acts(&c.reversed[g], x, z) {
                    gates.push(g);
                }
            });
        }
        if !gates.is_empty() {
            gates.sort_unstable();
            acted.push((*key, std::mem::take(terms), gates));
            return false;
        }
        if pass.fresh && !truncate(key, terms) {
            return false;
        }
        mask_or_assign(&mut stayed_active, &support);
        true
    });

    let mut outgoing: Vec<Emitted<NW>> = (0..pass.n_shards).map(|_| Vec::new()).collect();
    let mut work: Vec<(PauliKey<NW>, Vec<Term>, usize)> = Vec::new();
    let mut emitted: Emitted<NW> = Vec::new();
    for (key, terms, gates) in acted {
        work.push((key, terms, 0));
        while let Some((key, mut terms, pos)) = work.pop() {
            if pos == gates.len() {
                outgoing[shard_of(&key, pass.n_shards)].push((key, terms));
                continue;
            }
            let g = gates[pos];
            if pass.fresh && pos == 0 && pass.first_gate != Some(g) {
                // An observable word the first gate step left alone: it
                // was truncated along with everything else after it.
                if !truncate(&key, &mut terms) {
                    continue;
                }
            }
            // DeadQubitPruner: word is dead if it has X/Y (x-bit set) on a
            // qubit that no remaining gate (incl. this one) touches again.
            if c.use_dead_qubit_pruner && !mask_subset(&key.0, &c.active_qubits_from[g]) {
                continue;
            }
            if c.use_xy_weight_pruner && !xy_reachable(&key.0, &c.gate_masks[g..]) {
                continue;
            }
            evolve_one_gate(&c.reversed[g], key.0, key.1, terms, mono, &mut emitted);
            for (out_key, mut out_terms) in emitted.drain(..) {
                if truncate(&out_key, &mut out_terms) {
                    work.push((out_key, out_terms, pos + 1));
                }
            }
        }
    }
    (outgoing, stayed_active)
}

/// Second half of a layer, for one shard: merge in the words every shard
/// produced for it. Returns their support.
fn merge_shard<const NW: usize>(shard: &mut WordMap<NW>, incoming: Vec<Emitted<NW>>) -> [u64; NW] {
    let mut active = [0u64; NW];
    for batch in incoming {
        for (key, terms) in batch {
            mask_or_assign(&mut active, &key.0);
            mask_or_assign(&mut active, &key.1);
            insert_or_extend(shard, key, terms);
        }
    }
    active
}

/// `f` applied to every item, on one scoped thread per item when `parallel`,
/// with the results in item order.
fn run_all<T: Send, R: Send>(items: Vec<T>, parallel: bool, f: impl Fn(T) -> R + Sync) -> Vec<R> {
    if !parallel {
        return items.into_iter().map(f).collect();
    }
    let f = &f;
    std::thread::scope(|s| {
        let handles: Vec<_> = items
            .into_iter()
            .map(|item| s.spawn(move || f(item)))
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
            .collect()
    })
}

/// Propagate one observable backwards through the circuit.
///
/// The words are split over `shards` by `shard_of`, one shard per thread,
/// and every thread appends to its own monomial arena. Each layer (see
/// `Layer`) is one pass: every shard moves out the words some gate in the
/// layer acts on, leaving the rest where they are; each moved-out word is
/// carried through the gates that act on it, in circuit order; and the
/// results are merged into whichever shard owns them. The result is the same
/// as applying one gate at a time to the whole map, with pruning and
/// truncation after every gate:
///
/// * Truncation (k1/k2/coefficient) depends only on a word's key and terms,
///   so it only has to run when a gate produces a word. The observable's own
///   words are truncated once, like everything else after the first gate
///   step that touches any of them.
/// * Both exact pruners are monotone along the circuit: a word rejected at
///   one gate would be rejected at every later gate too (DeadQubitPruner's
///   set of still-reachable qubits only shrinks; XYWeightPruner's budget
///   only shrinks as gates are used up). So a word is checked only when a
///   gate is about to act on it, which rejects it exactly when some earlier
///   check would have. A dead word that no gate touches again just sits in
///   the map; it has an X/Y somewhere, so it never reaches the expectation.
fn heisenberg_one<const NW: usize>(
    circuit: &Circuit<NW>,
    layers: &[Layer<NW>],
    mut shards: Vec<WordMap<NW>>,
    arenas: &mut [Monomials],
) -> Vec<CoeffTerm> {
    let n_shards = shards.len();
    let mut active_mask = [0u64; NW];
    for (x, z) in shards.iter().flat_map(|s| s.keys()) {
        mask_or_assign(&mut active_mask, x);
        mask_or_assign(&mut active_mask, z);
    }
    // The first gate the one-gate-at-a-time loop would not skip. Until it
    // has been applied, the observable's own words are untruncated.
    let first_gate = circuit
        .gate_masks
        .iter()
        .position(|gm| mask_intersects(gm, &active_mask));

    let mut gate_on_qubit = vec![usize::MAX; NW * 64];
    let mut compact_at = MIN_COMPACT_NODES;
    let mut started = false;

    for layer in layers {
        if !mask_intersects(&layer.mask, &active_mask) {
            continue;
        }
        // Only the first layer processed still holds untruncated words.
        let fresh = !started;
        started = true;
        let layer_masks = &circuit.gate_masks[layer.start..layer.end];
        for (g, gm) in (layer.start..).zip(layer_masks) {
            for_each_bit(gm, |q| gate_on_qubit[q] = g);
        }

        let n_words: usize = shards.iter().map(|s| s.len()).sum();
        let parallel = n_shards > 1 && n_words >= PARALLEL_MIN_WORDS;
        let pass = LayerPass {
            circuit,
            layer,
            gate_on_qubit: &gate_on_qubit,
            fresh,
            first_gate,
            n_shards,
        };
        let evolved = run_all(
            shards.iter_mut().zip(arenas.iter_mut()).collect(),
            parallel,
            |(shard, mono)| evolve_shard(&pass, shard, mono),
        );
        let mut next_active = [0u64; NW];
        let mut incoming: Vec<Vec<Emitted<NW>>> = (0..n_shards).map(|_| Vec::new()).collect();
        for (outgoing, stayed_active) in evolved {
            mask_or_assign(&mut next_active, &stayed_active);
            for (dest, batch) in outgoing.into_iter().enumerate() {
                incoming[dest].push(batch);
            }
        }
        let merged = run_all(
            shards.iter_mut().zip(incoming).collect(),
            parallel,
            |(shard, batches)| merge_shard(shard, batches),
        );
        for mask in &merged {
            mask_or_assign(&mut next_active, mask);
        }
        active_mask = next_active;

        for gm in layer_masks {
            for_each_bit(gm, |q| gate_on_qubit[q] = usize::MAX);
        }
        let n_nodes: usize = arenas.iter().map(|a| a.len()).sum();
        if n_nodes >= compact_at || arenas.iter().any(|a| a.len() >= ARENA_CAPACITY / 2) {
            compact(arenas, &mut shards, n_shards > 1);
            let n_nodes: usize = arenas.iter().map(|a| a.len()).sum();
            compact_at = MIN_COMPACT_NODES.max(4 * n_nodes);
        }
    }

    to_expectation(&shards, arenas, n_shards > 1)
}

/// Call `f` with the index of every set bit in `mask`, lowest first.
#[inline]
fn for_each_bit<const NW: usize>(mask: &[u64; NW], mut f: impl FnMut(usize)) {
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            f(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn propagate_batch_impl<const NW: usize>(
    gate_kind: Vec<u8>,
    gate_wire0: Vec<u32>,
    gate_wire1: Vec<i64>,
    gate_param: Vec<i64>,
    gate_fixed: Vec<Option<f64>>,
    k1: i64,
    k2: i64,
    coeff_threshold: f64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
    paulidicts: Vec<Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>>,
    n_threads: usize,
) -> PyResult<Vec<Vec<CoeffTerm>>> {
    let n_gates = gate_kind.len();
    let mut gates: Vec<GateSpec<NW>> = Vec::with_capacity(n_gates);
    for i in 0..n_gates {
        let mut wire_mask = [0u64; NW];
        set_wire_bit(&mut wire_mask, gate_wire0[i]);
        if gate_wire1[i] >= 0 {
            set_wire_bit(&mut wire_mask, gate_wire1[i] as u32);
        }
        gates.push(GateSpec {
            kind: gate_kind[i],
            wire0: gate_wire0[i],
            wire1: gate_wire1[i],
            param: gate_param[i],
            fixed: gate_fixed[i],
            wire_mask,
        });
    }
    // Heisenberg picture: consume gates in reverse circuit order.
    let reversed: Vec<GateSpec<NW>> = gates.into_iter().rev().collect();
    let n = reversed.len();
    let gate_masks: Vec<[u64; NW]> = reversed.iter().map(|g| g.wire_mask).collect();
    let layers = build_layers(&gate_masks);

    // Suffix union of wire masks: active_qubits_from[i] = qubits touched by
    // gates i..n-1 (used by DeadQubitPruner).
    let mut active_qubits_from = vec![[0u64; NW]; n + 1];
    for i in (0..n).rev() {
        active_qubits_from[i] = mask_or(&active_qubits_from[i + 1], &reversed[i].wire_mask);
    }

    let circuit = Circuit {
        reversed: &reversed,
        gate_masks: &gate_masks,
        active_qubits_from: &active_qubits_from,
        k1,
        k2,
        coeff_threshold,
        use_dead_qubit_pruner,
        use_xy_weight_pruner,
    };
    let n_shards = n_threads.clamp(1, MAX_ARENAS);

    let mut results = Vec::with_capacity(paulidicts.len());
    for pd_spec in paulidicts {
        let mut arenas: Vec<Monomials> = (0..n_shards).map(Monomials::new).collect();
        let mut shards: Vec<WordMap<NW>> = (0..n_shards).map(|_| WordMap::default()).collect();
        for (xw, zw, coeff, sin_idx, cos_idx) in pd_spec {
            let key = (words_to_array::<NW>(&xw), words_to_array::<NW>(&zw));
            let term = arenas[0].term_from_lists(coeff, &sin_idx, &cos_idx);
            shards[shard_of(&key, n_shards)]
                .entry(key)
                .or_default()
                .push(term);
        }
        results.push(heisenberg_one(&circuit, &layers, shards, &mut arenas));
    }

    Ok(results)
}

/// Dispatches to the `$f::<NW>` monomorphization matching `$nw` (a
/// `words_needed(num_qubits)` result), or returns a `PyValueError` if `$nw`
/// exceeds every supported size. Add a literal here (and nowhere else) to
/// raise the qubit ceiling.
///
/// `$f` is a bare function path (not a call), and the numeric literal is
/// substituted directly into each arm's turbofish rather than bound to a
/// shared identifier - macro hygiene would make a `const NW` defined here
/// invisible to a `NW` token written at the call site, so each arm must
/// spell its word count out explicitly.
macro_rules! dispatch_nw {
    ($nw:expr, $num_qubits:expr, $f:ident, ( $($args:expr),* $(,)? )) => {
        match $nw {
            1 => $f::<1>($($args),*),
            2 => $f::<2>($($args),*),
            4 => $f::<4>($($args),*),
            8 => $f::<8>($($args),*),
            16 => $f::<16>($($args),*),
            32 => $f::<32>($($args),*),
            64 => $f::<64>($($args),*),
            128 => $f::<128>($($args),*),
            other => return Err(PyValueError::new_err(format!(
                "num_qubits={} needs {} u64 words per Pauli-word plane, which exceeds \
                 the maximum supported 128 words (8192 qubits) in this build of pprop_rs. \
                 Add a larger literal to dispatch_nw! in lib.rs if you need more.",
                $num_qubits, other
            ))),
        }
    };
}

#[pyfunction]
#[pyo3(signature = (
    num_qubits,
    gate_kind, gate_wire0, gate_wire1, gate_param, gate_fixed,
    k1, k2, coeff_threshold,
    use_dead_qubit_pruner, use_xy_weight_pruner,
    paulidicts,
    n_threads = 1,
))]
#[allow(clippy::too_many_arguments)]
fn propagate_batch(
    py: Python<'_>,
    num_qubits: u32,
    gate_kind: Vec<u8>,
    gate_wire0: Vec<u32>,
    gate_wire1: Vec<i64>,
    gate_param: Vec<i64>,
    gate_fixed: Vec<Option<f64>>,
    k1: i64,
    k2: i64,
    coeff_threshold: f64,
    use_dead_qubit_pruner: bool,
    use_xy_weight_pruner: bool,
    paulidicts: Vec<Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>>,
    n_threads: usize,
) -> PyResult<Vec<Vec<(f64, Vec<u32>, Vec<u32>)>>> {
    let nw = words_needed(num_qubits);
    py.allow_threads(move || {
        dispatch_nw!(
            nw, num_qubits, propagate_batch_impl,
            (
                gate_kind, gate_wire0, gate_wire1, gate_param, gate_fixed,
                k1, k2, coeff_threshold, use_dead_qubit_pruner, use_xy_weight_pruner,
                paulidicts, n_threads,
            )
        )
    })
}

fn evolve_single_gate_debug_impl<const NW: usize>(
    kind: u8,
    wire0: u32,
    wire1: i64,
    param: i64,
    fixed: Option<f64>,
    x: &[u64],
    z: &[u64],
    terms: Vec<(f64, Vec<u32>, Vec<u32>)>,
) -> Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)> {
    let mut wire_mask = [0u64; NW];
    set_wire_bit(&mut wire_mask, wire0);
    if wire1 >= 0 {
        set_wire_bit(&mut wire_mask, wire1 as u32);
    }
    let gate = GateSpec::<NW> { kind, wire0, wire1, param, fixed, wire_mask };
    let mut mono = Monomials::new(0);
    let terms: Vec<Term> = terms.iter().map(|(c, s, cc)| mono.term_from_lists(*c, s, cc)).collect();
    let mut out: Emitted<NW> = Vec::new();
    evolve_one_gate(&gate, words_to_array::<NW>(x), words_to_array::<NW>(z), terms, &mut mono, &mut out);
    let mut rows = Vec::new();
    for ((ox, oz), terms) in out {
        for t in terms {
            let (s, cc) = to_lists(std::slice::from_ref(&mono), &t);
            rows.push((ox.to_vec(), oz.to_vec(), t.coeff, s, cc));
        }
    }
    rows
}

/// Test-only helper: evolve a single Pauli word through one gate and return
/// the *unfiltered* resulting map (no zerobracket filtering, unlike
/// `propagate_batch`). Used by `tests/test_rule_tables.py` to cross-check
/// every gate's `rule` dict in `pprop/gates/*.py` directly against this
/// file's rule tables, term-for-term, independent of whether the output
/// happens to land in the Z/I subspace. `fixed` is the constant angle of a
/// non-trainable rotation (see `GateSpec::fixed`); leave it `None` for a
/// trainable one.
#[pyfunction]
#[pyo3(signature = (num_qubits, kind, wire0, wire1, param, x, z, terms, fixed = None))]
#[allow(clippy::too_many_arguments)]
fn evolve_single_gate_debug(
    num_qubits: u32,
    kind: u8,
    wire0: u32,
    wire1: i64,
    param: i64,
    x: Vec<u64>,
    z: Vec<u64>,
    terms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    fixed: Option<f64>,
) -> PyResult<Vec<(Vec<u64>, Vec<u64>, f64, Vec<u32>, Vec<u32>)>> {
    let nw = words_needed(num_qubits);
    Ok(dispatch_nw!(
        nw, num_qubits, evolve_single_gate_debug_impl,
        (kind, wire0, wire1, param, fixed, &x, &z, terms)
    ))
}

/// The CSR layout `pprop.propagator.evaluator.build_ragged_arrays` builds
/// (coefficients, lookup-table index of every factor, factors per term), as
/// native-endian float64/int64 buffers for `numpy.frombuffer`. Same layout,
/// entry for entry; this just skips the per-factor Python loop.
#[pyfunction]
fn ragged_layout<'py>(
    py: Python<'py>,
    expr: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_params: u32,
) -> (
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
    Bound<'py, PyByteArray>,
) {
    let sentinel = num_params as i64;
    let cos_offset = num_params as i64 + 1;
    let mut coeffs: Vec<u8> = Vec::with_capacity(8 * expr.len());
    let mut cnt: Vec<u8> = Vec::with_capacity(8 * expr.len());
    let mut idx: Vec<u8> = Vec::new();
    for (coeff, sin_idx, cos_idx) in &expr {
        coeffs.extend_from_slice(&coeff.to_ne_bytes());
        for &j in sin_idx {
            idx.extend_from_slice(&(j as i64).to_ne_bytes());
        }
        for &j in cos_idx {
            idx.extend_from_slice(&(cos_offset + j as i64).to_ne_bytes());
        }
        let mut n = (sin_idx.len() + cos_idx.len()) as i64;
        if n == 0 {
            idx.extend_from_slice(&sentinel.to_ne_bytes());
            n = 1;
        }
        cnt.extend_from_slice(&n.to_ne_bytes());
    }
    (
        PyByteArray::new_bound(py, &coeffs),
        PyByteArray::new_bound(py, &idx),
        PyByteArray::new_bound(py, &cnt),
    )
}

/// Add `term` to the accumulator slot of every factor in `run`, spreading
/// consecutive factors over `C` interleaved copies of the accumulator so that
/// a recurring parameter does not stall on the previous read-modify-write.
///
/// The writes skip bounds checking: `Evaluator::new` rejects any index that
/// does not address one copy, and `acc` is `C * width` long.
#[inline]
fn scatter<const C: usize>(run: &[u32], term: f64, acc: &mut [f64], width: usize) {
    let mut chunks = run.chunks_exact(C);
    for chunk in &mut chunks {
        for (copy, &entry) in chunk.iter().enumerate() {
            unsafe { *acc.get_unchecked_mut(copy * width + entry as usize) += term };
        }
    }
    for &entry in chunks.remainder() {
        unsafe { *acc.get_unchecked_mut(entry as usize) += term };
    }
}

/// Compiled evaluator for one propagated expression, mirroring the NumPy
/// implementation in `pprop.propagator.evaluator` (see `build_ragged_arrays`
/// there for how `coeffs`/`idx`/`cnt` are produced and what the index space
/// means). Everything below is that same arithmetic with the intermediate
/// arrays removed: the terms are walked once, factor by factor, keeping the
/// running products in registers rather than materialising one temporary per
/// pass.
///
/// Index space, with `P = num_params`: a factor `sin(theta_k)` is `k`, a
/// factor `cos(theta_k)` is `P + 1 + k`, and `P` itself is a sentinel whose
/// value is 1 (it stands in for a term with no factors left). Powers are
/// repeated indices, so no exponent is ever evaluated.
#[pyclass]
pub struct Evaluator {
    coeffs: Vec<f64>,
    idx: Vec<u32>,
    /// Start of each term's run in `idx`, with a closing entry: length is
    /// `coeffs.len() + 1`.
    off: Vec<usize>,
    num_params: usize,
    /// Longest single term, i.e. the scratch space the exact path needs.
    max_run: usize,
    /// How many interleaved copies of the gradient accumulator to keep -
    /// 4, 2 or 1, whichever fits in L1 for this parameter count. See
    /// `scatter`.
    acc_copies: usize,
    tol: f64,
}

impl Evaluator {
    /// `[sin(theta_0..P-1), 1.0, cos(theta_0..P-1)]`, the table `idx` reads.
    fn table(&self, sins: &[f64], coss: &[f64]) -> Vec<f64> {
        let p = self.num_params;
        let mut table = Vec::with_capacity(2 * p + 1);
        table.extend_from_slice(sins);
        table.push(1.0);
        table.extend_from_slice(coss);
        table
    }

    /// Product of one term's factors. Four independent accumulators, so the
    /// multiplies pipeline instead of forming one dependency chain per term.
    ///
    /// The table lookups skip bounds checking: `new` rejects any index that
    /// does not address `table`, which is the only place `idx` is set.
    #[inline]
    fn term_product(&self, table: &[f64], start: usize, end: usize) -> f64 {
        debug_assert!(table.len() == 2 * self.num_params + 1);
        let run = &self.idx[start..end];
        let (mut a, mut b, mut c, mut d) = (1.0, 1.0, 1.0, 1.0);
        let mut chunks = run.chunks_exact(4);
        for chunk in &mut chunks {
            unsafe {
                a *= *table.get_unchecked(chunk[0] as usize);
                b *= *table.get_unchecked(chunk[1] as usize);
                c *= *table.get_unchecked(chunk[2] as usize);
                d *= *table.get_unchecked(chunk[3] as usize);
            }
        }
        for &entry in chunks.remainder() {
            a *= unsafe { *table.get_unchecked(entry as usize) };
        }
        (a * b) * (c * d)
    }

    /// Gradient via the logarithmic derivative: every factor on parameter `k`
    /// contributes `term * cot(theta_k)` (or `-term * tan(theta_k)` for a
    /// cosine), which depends only on `k`. So the factors just accumulate
    /// term values per parameter and the cot/tan multiply happens once over
    /// the parameter vector. Requires every angle to sit away from a zero of
    /// sin/cos; `grad_exact` handles the rest.
    fn grad_fast(&self, sins: &[f64], coss: &[f64], grad: &mut [f64]) -> f64 {
        let p = self.num_params;
        let table = self.table(sins, coss);
        let width = 2 * p + 1;
        // Consecutive factors are scattered into *different* copies of the
        // accumulator, so a parameter that recurs within a few factors does
        // not stall waiting on the previous read-modify-write. The copies are
        // summed back together at the end.
        let copies = self.acc_copies;
        let mut acc = vec![0.0f64; copies * width];
        let mut value = 0.0;

        for t in 0..self.coeffs.len() {
            let (start, end) = (self.off[t], self.off[t + 1]);
            let term = self.coeffs[t] * self.term_product(&table, start, end);
            value += term;
            let run = &self.idx[start..end];
            match copies {
                4 => scatter::<4>(run, term, &mut acc, width),
                2 => scatter::<2>(run, term, &mut acc, width),
                _ => scatter::<1>(run, term, &mut acc, width),
            }
        }

        for k in 0..p {
            let mut sin_total = 0.0;
            let mut cos_total = 0.0;
            for c in 0..copies {
                sin_total += acc[c * width + k];
                cos_total += acc[c * width + p + 1 + k];
            }
            grad[k] = (coss[k] / sins[k]) * sin_total - (sins[k] / coss[k]) * cos_total;
        }
        value
    }

    /// Gradient without dividing by anything: for each term, walk its factors
    /// forwards recording partial products, then backwards multiplying by the
    /// running suffix, which gives the product of all *other* factors exactly.
    /// Costs a second pass over each term and some scratch, and is used only
    /// when an angle sits at (or extremely near) a zero of sin or cos, where
    /// the cot/tan form above is singular. Repeated indices come out right on
    /// their own: p copies each contribute the product of the other p-1.
    fn grad_exact(&self, sins: &[f64], coss: &[f64], grad: &mut [f64]) -> f64 {
        let p = self.num_params;
        let table = self.table(sins, coss);
        let mut prefix = vec![0.0f64; self.max_run];
        let mut value = 0.0;

        for t in 0..self.coeffs.len() {
            let (start, end) = (self.off[t], self.off[t + 1]);
            let coeff = self.coeffs[t];

            let mut running = 1.0;
            for (j, i) in (start..end).enumerate() {
                prefix[j] = running;
                running *= table[self.idx[i] as usize];
            }
            value += coeff * running;

            let mut suffix = 1.0;
            for (j, i) in (start..end).enumerate().rev() {
                let entry = self.idx[i] as usize;
                let excluding = coeff * prefix[j] * suffix;
                if entry < p {
                    grad[entry] += excluding * coss[entry];        // d sin / d theta
                } else if entry > p {
                    let k = entry - (p + 1);
                    grad[k] -= excluding * sins[k];                // d cos / d theta
                }
                suffix *= table[entry];
            }
        }
        value
    }
}

#[pymethods]
impl Evaluator {
    /// `coeffs`, `idx` and `cnt` are the arrays `build_ragged_arrays` returns,
    /// as buffers (`float64`, `uint32`, `uint32`).
    #[new]
    #[pyo3(signature = (coeffs, idx, cnt, num_params, tol = 1e-6))]
    fn new(
        coeffs: &Bound<'_, PyAny>,
        idx: &Bound<'_, PyAny>,
        cnt: &Bound<'_, PyAny>,
        num_params: usize,
        tol: f64,
    ) -> PyResult<Self> {
        let coeffs = read_f64(coeffs)?;
        let idx = read_u32(idx)?;
        let cnt = read_u32(cnt)?;
        if cnt.len() != coeffs.len() {
            return Err(PyValueError::new_err("coeffs and cnt must have equal length"));
        }

        let mut off = Vec::with_capacity(cnt.len() + 1);
        let mut total = 0usize;
        off.push(0);
        for &c in &cnt {
            total += c as usize;
            off.push(total);
        }
        if total != idx.len() {
            return Err(PyValueError::new_err("cnt does not add up to len(idx)"));
        }
        let limit = 2 * num_params + 1;
        if idx.iter().any(|&i| i as usize >= limit) {
            return Err(PyValueError::new_err("factor index out of range"));
        }

        let max_run = cnt.iter().map(|&c| c as usize).max().unwrap_or(0);
        // Keep the interleaved accumulators inside ~16 KB, i.e. half of a
        // typical L1, or the trick costs more in misses than it saves. Wide
        // parameter vectors settle for two copies, or one.
        let width = 2 * num_params + 1;
        let bytes = width * std::mem::size_of::<f64>();
        let acc_copies = [4, 2, 1].into_iter().find(|c| c * bytes <= 16 * 1024).unwrap_or(1);
        Ok(Evaluator { coeffs, idx, off, num_params, max_run, acc_copies, tol })
    }

    /// Expectation value at the given `sin(theta)`/`cos(theta)`.
    fn eval(&self, py: Python<'_>, sins: &Bound<'_, PyAny>, coss: &Bound<'_, PyAny>)
        -> PyResult<f64>
    {
        let sins = self.read_angles(sins)?;
        let coss = self.read_angles(coss)?;
        Ok(py.allow_threads(|| {
            let table = self.table(&sins, &coss);
            (0..self.coeffs.len())
                .map(|t| self.coeffs[t] * self.term_product(&table, self.off[t], self.off[t + 1]))
                .sum()
        }))
    }

    /// Expectation value, writing the gradient into `out` (`float64`, length
    /// `num_params`) rather than allocating a fresh array per call.
    fn eval_and_grad(
        &self,
        py: Python<'_>,
        sins: &Bound<'_, PyAny>,
        coss: &Bound<'_, PyAny>,
        out: &Bound<'_, PyAny>,
    ) -> PyResult<f64> {
        let sins = self.read_angles(sins)?;
        let coss = self.read_angles(coss)?;
        let buffer = PyBuffer::<f64>::get_bound(out)?;
        if buffer.item_count() != self.num_params || buffer.readonly() {
            return Err(PyValueError::new_err("out must be a writable float64 array of length num_params"));
        }

        let mut grad = vec![0.0f64; self.num_params];
        let value = py.allow_threads(|| {
            let singular = sins.iter().chain(coss.iter()).any(|v| v.abs() < self.tol);
            if singular {
                self.grad_exact(&sins, &coss, &mut grad)
            } else {
                self.grad_fast(&sins, &coss, &mut grad)
            }
        });
        buffer.copy_from_slice(py, &grad)?;
        Ok(value)
    }
}

impl Evaluator {
    fn read_angles(&self, obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
        let values = read_f64(obj)?;
        if values.len() != self.num_params {
            return Err(PyValueError::new_err("sins/coss must have length num_params"));
        }
        Ok(values)
    }
}

fn read_f64(obj: &Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    PyBuffer::<f64>::get_bound(obj)?.to_vec(obj.py())
}

fn read_u32(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    PyBuffer::<u32>::get_bound(obj)?.to_vec(obj.py())
}

#[pymodule]
fn pprop_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(propagate_batch, m)?)?;
    m.add_function(wrap_pyfunction!(ragged_layout, m)?)?;
    m.add_class::<Evaluator>()?;
    m.add_function(wrap_pyfunction!(evolve_single_gate_debug, m)?)?;
    Ok(())
}
