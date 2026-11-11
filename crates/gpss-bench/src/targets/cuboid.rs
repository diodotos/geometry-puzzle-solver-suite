//! Generic `AxBxC` cuboids: the 2x2x2 and 3x3x2 cases of the shared
//! `conformance/corpus/cuboid.jsonl` (exact single-layer-metric optima from a BFS ball),
//! one workload per size, solved by uninformed IDA* up to depth 4.

use gpss_core::search::IdaStar;
use gpss_core::ZeroHeuristic;
use gpss_puzzles::cuboid::{Cuboid, Dims};

use crate::corpus::BucketScheme;
use crate::target::{Registry, SolverSpec, Workload};

/// Short 1-2, mid 3, top 4 (the deepest exact cases in scope).
pub const BUCKETS: BucketScheme = BucketScheme::new(2, 3);

/// Sizes registered, as they appear in the corpus `puzzle` field.
pub const SIZES: [&str; 2] = ["2x2x2", "3x3x2"];

/// Register one `cuboid-<size>` workload per entry of [`SIZES`].
pub fn register(registry: &mut Registry) {
    for size in SIZES {
        let dims: Dims = size.parse().expect("valid cuboid size");
        let workload = Workload::new(
            Cuboid::new(dims),
            format!("cuboid-{size}"),
            BUCKETS,
            |cuboid: &Cuboid, scramble: &str| {
                let moves = cuboid.parse_sequence(scramble).map_err(|e| e.to_string())?;
                Ok(cuboid.scrambled(&moves))
            },
        )
        .corpus_subset("cuboid", size);
        let solvers = vec![SolverSpec::new("idastar-zero", |_: &Cuboid| {
            Box::new(IdaStar::new(ZeroHeuristic))
        })
        .max_depth(4)];
        registry.add(workload, solvers);
    }
}
