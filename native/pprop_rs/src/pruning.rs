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
