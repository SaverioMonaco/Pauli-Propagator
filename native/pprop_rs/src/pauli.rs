//! Pauli words as bit arrays: one bit per qubit in an x plane and a z plane.
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

// (x, z) bitmask planes, same convention as pprop.pauli.op.PauliOp, packed
// into NW little-endian u64 words instead of one u64 (see module docs above
// for the worked "Y0 Z5 X70" example). A PauliKey<NW> value fully identifies
// one Pauli word; FxHashMap<PauliKey<NW>, Vec<Term>> is "one Pauli word
// -> its current coefficient", the data structure the whole propagation
// loop below updates at every gate step.
pub(crate) type PauliKey<const NW: usize> = ([u64; NW], [u64; NW]);

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
pub(crate) fn words_needed(num_qubits: u32) -> usize {
    let raw = ((num_qubits as usize) + 63) / 64;
    raw.max(1).next_power_of_two()
}

#[inline]
pub(crate) fn word_bit(wire: u32) -> (usize, u32) {
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
pub(crate) fn words_to_array<const NW: usize>(words: &[u64]) -> [u64; NW] {
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
pub(crate) fn pauli_label<const NW: usize>(x: &[u64; NW], z: &[u64; NW], wire: u32) -> u8 {
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
pub(crate) fn set_label<const NW: usize>(
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
pub(crate) fn set_wire_bit<const NW: usize>(arr: &mut [u64; NW], wire: u32) {
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
pub(crate) fn mask_intersects<const NW: usize>(a: &[u64; NW], b: &[u64; NW]) -> bool {
    a.iter().zip(b.iter()).any(|(x, y)| x & y != 0)
}

// In-place "OR every word of b into a" (mutates a, nothing returned) -
// `&mut` like set_wire_bit earlier, same reasoning. Used to accumulate a
// running union of many separate masks into one existing array without
// allocating a new one each time.
pub(crate) fn mask_or_assign<const NW: usize>(a: &mut [u64; NW], b: &[u64; NW]) {
    for i in 0..NW {
        a[i] |= b[i];
    }
}

// Same "OR every word together" idea as mask_or_assign, but builds and
// returns a brand new array instead of mutating an existing one - used
// where the caller needs to keep `a` unchanged (e.g. building a table of
// several different union results side by side, each one immutable once
// computed).
pub(crate) fn mask_or<const NW: usize>(a: &[u64; NW], b: &[u64; NW]) -> [u64; NW] {
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
pub(crate) fn mask_popcount<const NW: usize>(a: &[u64; NW]) -> u32 {
    a.iter().map(|w| w.count_ones()).sum()
}

// Total Pauli weight: popcount(x | z) summed across words - every qubit
// that's X, Y, *or* Z counts (unlike mask_popcount above, which only
// counted X/Y). `.zip(z.iter())` pairs up x[i] with z[i] word-by-word,
// `(a | b).count_ones()` combines and counts each pair, `.sum()` adds
// those per-word counts into one total.
pub(crate) fn combined_weight<const NW: usize>(x: &[u64; NW], z: &[u64; NW]) -> u32 {
    x.iter().zip(z.iter()).map(|(a, b)| (a | b).count_ones()).sum()
}

// "Is every 1-bit of `a` also a 1-bit of `allowed`?" (a is a subset of
// allowed, as bit sets). Per word: `!al` flips `allowed`'s bits (every bit
// NOT allowed), `x & !al` keeps only the bits of `x` that fall outside
// `allowed`, and `== 0` checks there are none - i.e. no bit of `x` lies
// outside `allowed`. `.all(...)` requires this true for every word, not
// just one. Used by DeadQubitPruner: a Pauli word survives only if every
// qubit where it carries X/Y is still touchable by some remaining gate.
pub(crate) fn mask_subset<const NW: usize>(a: &[u64; NW], allowed: &[u64; NW]) -> bool {
    a.iter().zip(allowed.iter()).all(|(x, al)| x & !al == 0)
}

// Is every word of `a` exactly 0 (the whole array represents "no qubit
// set")? `*w` dereferences `w` (a `&u64` from `.iter()`) back to a plain
// `u64` to compare against `0` - `.iter()` yields references, and `==`
// needs values on the same footing on both sides. This is the NW-word
// generalization of the old single-word `x == 0` check, used by
// `to_expectation` to test whether a Pauli word is entirely Z/I (its
// x-plane is all zero) and therefore contributes to <0|P|0>.
pub(crate) fn mask_is_zero<const NW: usize>(a: &[u64; NW]) -> bool {
    a.iter().all(|w| *w == 0)
}

/// Call `f` with the index of every set bit in `mask`, lowest first.
#[inline]
pub(crate) fn for_each_bit<const NW: usize>(mask: &[u64; NW], mut f: impl FnMut(usize)) {
    for (w, &word) in mask.iter().enumerate() {
        let mut bits = word;
        while bits != 0 {
            f(w * 64 + bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
    }
}
