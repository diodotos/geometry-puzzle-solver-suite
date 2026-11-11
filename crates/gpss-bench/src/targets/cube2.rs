//! 2x2x2 workload: the 233-case `conformance/corpus/cube2.jsonl` corpus, buckets short
//! 1-5 / mid 6-9 / top 10-11 (the generator's tags; the diameter is 11 in the half-turn
//! metric), and the solver configurations the CLI exposes.

use gpss_core::pdb::PatternDatabase;
use gpss_core::search::{AStar, Bfs, IdaStar};
use gpss_puzzles::cube::parse_sequence;
use gpss_puzzles::cube2::{Cube2, Cube2Heuristic, FullCoord, OrientationCoord, PermutationCoord};

use crate::corpus::BucketScheme;
use crate::target::{Registry, SolverSpec, Workload};

/// Depth ranges matching the corpus generator's `short` / `mid` / `top` tags.
pub const BUCKETS: BucketScheme = BucketScheme::new(5, 9);

/// Register the 2x2x2 and its solvers.
pub fn register(registry: &mut Registry) {
    let mut workload = Workload::new(
        Cube2::new(),
        "cube2",
        BUCKETS,
        |cube: &Cube2, scramble: &str| {
            let moves = parse_sequence(scramble).map_err(|e| e.to_string())?;
            // Search runs in the fixed DBL frame (ADR 0004); the regripped state is the start.
            Ok(cube.regrip(&cube.scrambled(&moves)).0)
        },
    );
    for h in Cube2Heuristic::ALL {
        workload = workload.heuristic(h.name(), move |c: &Cube2| h.build(c));
    }
    let workload = workload
        .table("orientation", |c: &Cube2| {
            PatternDatabase::build(c, &OrientationCoord)
        })
        .table("permutation", |c: &Cube2| {
            PatternDatabase::build(c, &PermutationCoord)
        })
        .table("perfect", |c: &Cube2| PatternDatabase::build(c, &FullCoord));

    let solvers = vec![
        // BFS takes ~0.1 s at depth 8 and ~2.5 s at depth 11; keep CI to depth <= 8.
        SolverSpec::new("bfs", |_: &Cube2| Box::new(Bfs)).ci_max_depth(8),
        SolverSpec::new("astar-dual", |c: &Cube2| {
            Box::new(AStar::new(Cube2Heuristic::Dual.build(c)))
        }),
        SolverSpec::new("astar-dual-w2", |c: &Cube2| {
            Box::new(AStar::weighted(Cube2Heuristic::Dual.build(c), 2.0))
        })
        .max_ratio(2.0),
        SolverSpec::new("idastar-dual", |c: &Cube2| {
            Box::new(IdaStar::new(Cube2Heuristic::Dual.build(c)))
        }),
        SolverSpec::new("idastar-perfect", |c: &Cube2| {
            Box::new(IdaStar::new(Cube2Heuristic::Perfect.build(c)))
        }),
    ];
    registry.add(workload, solvers);
}
