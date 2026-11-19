//! Exact pruners: drop Pauli words that can never reach the readout.

use crate::pauli::*;

/// XYWeightPruner verdict: can the gates in `gate_masks` (the rest of the
/// circuit, in order) still reduce this word's XY-weight to zero? Each gate
/// that touches the growing causal cone of `x` can remove at most one X/Y.
pub(crate) fn xy_reachable<const NW: usize>(x: &[u64; NW], gate_masks: &[[u64; NW]]) -> bool {
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

/// For the DeadQubitPruner: `masks[i]` = the qubits touched by gates
/// `i..n-1` of `gate_masks` (the gates in propagation order, i.e. the circuit
/// reversed), with `masks[n]` empty. A word with X/Y on a qubit outside
/// `masks[i]` can never lose it once gate `i` is reached.
pub(crate) fn dead_qubit_masks<const NW: usize>(gate_masks: &[[u64; NW]]) -> Vec<[u64; NW]> {
    let n = gate_masks.len();
    let mut masks = vec![[0u64; NW]; n + 1];
    for i in (0..n).rev() {
        masks[i] = mask_or(&masks[i + 1], &gate_masks[i]);
    }
    masks
}
