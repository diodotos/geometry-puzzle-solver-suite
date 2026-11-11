//! Wall-clock timing of the 3x3x3 solvers on fixed-seed scrambles, used for the
//! before/after tables in `docs/theory/kociemba-two-phase.md` and
//! `docs/theory/korf-optimal-3x3.md`.
//!
//! ```text
//! cargo run --release -p gpss-puzzles --example cube3_timing -- twophase [N]
//! cargo run --release -p gpss-puzzles --example cube3_timing -- korf [LEN...]
//! ```
//!
//! `twophase` solves the superflip and `N` (default 200) random 40-move scrambles
//! (seed 1). `korf` solves 8 random scrambles of each length (default 10 11 12 13,
//! seed 2) optimally; `KORF_LOOKUPS=plain` or `KORF_LOOKUPS=0i,5,5i` (symmetry
//! index, `i` for the inverse) overrides the heuristic's extra lookups. Table setup (build or cache load) is timed separately. The
//! machine and the cache state should be reported with any numbers taken from here.

use std::time::Instant;

use gpss_core::search::{SearchLimits, Solution};
use gpss_puzzles::cube::{parse_sequence, random_scramble, FaceTurn};
use gpss_puzzles::cube3::{korf, Cube3, TwoPhase};
use rand::{rngs::StdRng, SeedableRng};

const SUPERFLIP: &str = "U R2 F B R B2 R U2 L B2 R U' D' R2 F R' L B2 U2 F2";

fn percentile(sorted: &[f64], p: f64) -> f64 {
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i]
}

fn check(cube: &Cube3, scramble: &[FaceTurn], sol: &Solution<FaceTurn>) {
    assert_eq!(
        cube.scrambled(scramble).then_all(&sol.moves),
        cube.scrambled(&[]),
        "invalid solution for {scramble:?}"
    );
}

fn twophase(n: usize) {
    let cube = Cube3::new();
    let t = Instant::now();
    TwoPhase::warm_up();
    println!("twophase setup: {:.3} s", t.elapsed().as_secs_f64());
    let solver = TwoPhase::default();
    let limits = SearchLimits::default();

    let sf = parse_sequence(SUPERFLIP).unwrap();
    let t = Instant::now();
    let sol = cube.solve_scramble(&sf, &solver, &limits).unwrap();
    check(&cube, &sf, &sol);
    println!(
        "superflip: {:.4} s, {} moves, {} expanded",
        t.elapsed().as_secs_f64(),
        sol.moves.len(),
        sol.stats.expanded
    );

    let mut rng = StdRng::seed_from_u64(1);
    let mut times = Vec::with_capacity(n);
    let (mut len_sum, mut len_max, mut expanded) = (0usize, 0usize, 0u64);
    for _ in 0..n {
        let scramble = random_scramble(&mut rng, 40);
        let t = Instant::now();
        let sol = cube.solve_scramble(&scramble, &solver, &limits).unwrap();
        times.push(t.elapsed().as_secs_f64());
        check(&cube, &scramble, &sol);
        len_sum += sol.moves.len();
        len_max = len_max.max(sol.moves.len());
        expanded += sol.stats.expanded;
    }
    let total: f64 = times.iter().sum();
    times.sort_by(f64::total_cmp);
    println!(
        "random x{n}: median {:.2} ms, mean {:.2} ms, p90 {:.2} ms, max {:.2} ms, \
         mean length {:.2}, max length {len_max}, mean expanded {:.0}",
        1e3 * percentile(&times, 0.5),
        1e3 * total / n as f64,
        1e3 * percentile(&times, 0.9),
        1e3 * times[n - 1],
        len_sum as f64 / n as f64,
        expanded as f64 / n as f64,
    );
}

fn korf(lens: &[usize]) {
    let cube = Cube3::new();
    let t = Instant::now();
    let heuristic = match std::env::var("KORF_LOOKUPS") {
        Ok(spec) if spec == "plain" => korf::KorfHeuristic::plain(),
        Ok(spec) => korf::KorfHeuristic::with_lookups(
            spec.split(',')
                .map(|l| korf::Lookup {
                    sym: l.trim_end_matches('i').parse().unwrap(),
                    inverse: l.ends_with('i'),
                })
                .collect(),
        ),
        Err(_) => korf::KorfHeuristic::new(),
    };
    let solver = gpss_core::search::IdaStar::new(heuristic);
    println!("korf setup: {:.3} s", t.elapsed().as_secs_f64());
    let limits = SearchLimits::default();
    let mut rng = StdRng::seed_from_u64(2);
    for &len in lens {
        let mut times = Vec::new();
        let (mut expanded, mut optimal) = (0u64, Vec::new());
        for _ in 0..8 {
            let scramble = random_scramble(&mut rng, len);
            let t = Instant::now();
            let sol = cube.solve_scramble(&scramble, &solver, &limits).unwrap();
            times.push(t.elapsed().as_secs_f64());
            check(&cube, &scramble, &sol);
            expanded += sol.stats.expanded;
            optimal.push(sol.moves.len());
        }
        let total: f64 = times.iter().sum();
        times.sort_by(f64::total_cmp);
        println!(
            "scramble length {len}: optimal lengths {optimal:?}, total {total:.2} s, \
             median {:.3} s, max {:.3} s, mean expanded {:.0}",
            percentile(&times, 0.5),
            times[times.len() - 1],
            expanded as f64 / 8.0,
        );
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("twophase") => twophase(args.get(1).map_or(200, |n| n.parse().unwrap())),
        Some("korf") => {
            let lens: Vec<usize> = args[1..].iter().map(|s| s.parse().unwrap()).collect();
            korf(if lens.is_empty() {
                &[10, 11, 12, 13]
            } else {
                &lens
            });
        }
        _ => eprintln!("usage: cube3_timing twophase [N] | korf [LEN...]"),
    }
}
