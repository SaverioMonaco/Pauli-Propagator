//! Splitting the work over threads.

use crate::pauli::PauliKey;
use rustc_hash::FxHasher;
use std::hash::{Hash, Hasher};

/// The shard (and so the thread) that owns `key`. The hash is remixed so
/// the keys within one shard still spread over all of its map's buckets.
pub(crate) fn shard_of<const NW: usize>(key: &PauliKey<NW>, n_shards: usize) -> usize {
    if n_shards == 1 {
        return 0;
    }
    let mut h = FxHasher::default();
    key.hash(&mut h);
    let mixed = h.finish().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32;
    ((mixed * n_shards as u64) >> 32) as usize
}

/// `f` applied to every item, on one scoped thread per item when `parallel`,
/// with the results in item order.
pub(crate) fn run_all<T: Send, R: Send>(items: Vec<T>, parallel: bool, f: impl Fn(T) -> R + Sync) -> Vec<R> {
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
