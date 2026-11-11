//! 3x3x3 workload: the `conformance/corpus/cube3.jsonl` corpus (exact optima for depths
//! 1-7 by meet-in-the-middle, superflip at 20, long random scrambles with unknown optima),
//! buckets short 1-5 / mid 6-7 / top >= 8 (the generator's tags), Kociemba two-phase and
//! Korf's optimal IDA*.

use gpss_puzzles::cube::parse_sequence;
use gpss_puzzles::cube3::{korf, Cube3, TwoPhase};

use crate::corpus::BucketScheme;
use crate::target::{Registry, SolverSpec, Workload};

/// Depth ranges matching the corpus generator's `short` / `mid` / `top` tags.
pub const BUCKETS: BucketScheme = BucketScheme::new(5, 7);

/// Register the 3x3x3 and its solvers.
pub fn register(registry: &mut Registry) {
    let workload = Workload::new(
        Cube3::new(),
        "cube3",
        BUCKETS,
        |cube: &Cube3, scramble: &str| {
            let moves = parse_sequence(scramble).map_err(|e| e.to_string())?;
            Ok(cube.scrambled(&moves))
        },
    );
    let solvers = vec![
        // Not optimal and no length ratio bound: it stops at the first solution of <= 20
        // moves, so only validity and "never shorter than the optimum" are checked. Its
        // counters are deterministic (no timeout is set), so they are baselined.
        SolverSpec::new("twophase", |_: &Cube3| {
            // The tables are process-global; build them here so setup cost is measured.
            TwoPhase::warm_up();
            Box::new(TwoPhase::new(20))
        }),
        // Optimal. The 87 MB tables are loaded from (or built into) `korf::cache_dir()`
        // in setup. Limited to the exact-depth cases: superflip and the random scrambles
        // would take minutes to hours.
        SolverSpec::new("idastar-korf", |_: &Cube3| Box::new(korf::solver())).max_depth(7),
    ];
    registry.add(workload, solvers);
}
