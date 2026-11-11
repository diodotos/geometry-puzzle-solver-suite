//! Kociemba's two-phase algorithm [Koc].
//!
//! Phase 1 searches (IDA*) for a move sequence taking the cube into the subgroup
//! `G1 = <U, D, R2, L2, F2, B2>`, where all orientations are solved and the UD-slice
//! edges are in the slice. Phase 2 solves inside `G1` using only those ten moves. Phase 1
//! solutions are enumerated in increasing length and each is completed by an optimal
//! phase 2, so the total length keeps improving; the search stops at the first solution
//! no longer than `target_length` (or at the limits, returning the best found so far).
//!
//! Heuristics are the maxima of two pruning tables per phase (twist x slice,
//! flip x slice and twist x flip; corner-perm x slice-perm and edge-perm x slice-perm), 1-5 MB each
//! and built in well under a second. Kociemba's own implementation additionally uses
//! symmetry reduction and larger tables; see `docs/theory/kociemba-two-phase.md`.

use super::coord::*;
use super::{Cube3, Cube3State};
use crate::cube::FaceTurn;
use gpss_core::search::{SearchError, SearchLimits, SearchStats, Solution, Solver};
use std::sync::OnceLock;
use std::time::Instant;

/// Indices (in `FaceTurn::all()` order) of the phase-2 moves `U U2 U' D D2 D' R2 L2 F2 B2`.
const P2_MOVES: [usize; 10] = [0, 1, 2, 9, 10, 11, 4, 13, 7, 16];

struct Tables {
    moves: Vec<FaceTurn>,
    follow: [[bool; 18]; 18],
    twist_mv: Vec<[u16; 18]>,
    flip_mv: Vec<[u16; 18]>,
    slice_mv: Vec<[u16; 18]>,
    cperm_mv: Vec<[u16; 10]>,
    udeperm_mv: Vec<[u16; 10]>,
    sperm_mv: Vec<[u8; 10]>,
    prune_twist_slice: Vec<u8>,
    prune_flip_slice: Vec<u8>,
    prune_twist_flip: Vec<u8>,
    prune_cperm_sperm: Vec<u8>,
    prune_udeperm_sperm: Vec<u8>,
}

fn move_table<T: Copy + Default, const M: usize>(
    n: usize,
    moves: &[usize; M],
    set: impl Fn(&mut Cube3State, usize),
    get: impl Fn(&Cube3State) -> T,
) -> Vec<[T; M]> {
    (0..n)
        .map(|i| {
            let mut s = Cube3State::SOLVED;
            set(&mut s, i);
            let mut row = [T::default(); M];
            for (j, &m) in moves.iter().enumerate() {
                row[j] = get(&s.then(&super::move_table()[m]));
            }
            row
        })
        .collect()
}

/// Breadth-first distances over a product coordinate `a * nb + b`.
fn prune_table(
    na: usize,
    nb: usize,
    start: usize,
    nmoves: usize,
    next: impl Fn(usize, usize, usize) -> (usize, usize),
) -> Vec<u8> {
    let mut table = vec![u8::MAX; na * nb];
    table[start] = 0;
    let mut frontier = vec![start];
    let mut depth = 0u8;
    while !frontier.is_empty() {
        let mut next_frontier = Vec::new();
        for &idx in &frontier {
            let (a, b) = (idx / nb, idx % nb);
            for m in 0..nmoves {
                let (a2, b2) = next(a, b, m);
                let j = a2 * nb + b2;
                if table[j] == u8::MAX {
                    table[j] = depth + 1;
                    next_frontier.push(j);
                }
            }
        }
        frontier = next_frontier;
        depth += 1;
    }
    table
}

fn tables() -> &'static Tables {
    static T: OnceLock<Tables> = OnceLock::new();
    T.get_or_init(|| {
        let moves: Vec<FaceTurn> = FaceTurn::all().collect();
        let mut follow = [[false; 18]; 18];
        for (i, &a) in moves.iter().enumerate() {
            for (j, &b) in moves.iter().enumerate() {
                follow[i][j] = FaceTurn::can_follow(a, b);
            }
        }
        let all: [usize; 18] = std::array::from_fn(|i| i);
        let twist_mv = move_table(N_TWIST, &all, |s, i| set_twist(s, i as u16), twist);
        let flip_mv = move_table(N_FLIP, &all, |s, i| set_flip(s, i as u16), flip);
        let slice_mv = move_table(N_SLICE, &all, |s, i| set_slice(s, i as u16), slice);
        let cperm_mv = move_table(N_CPERM, &P2_MOVES, |s, i| set_cperm(s, i as u16), cperm);
        let udeperm_mv = move_table(
            N_UDEPERM,
            &P2_MOVES,
            |s, i| set_udeperm(s, i as u16),
            udeperm,
        );
        let sperm_mv = move_table(N_SPERM, &P2_MOVES, |s, i| set_sperm(s, i as u8), sperm);
        let ss = SLICE_SOLVED as usize;
        let prune_twist_slice = prune_table(N_TWIST, N_SLICE, ss, 18, |a, b, m| {
            (twist_mv[a][m] as usize, slice_mv[b][m] as usize)
        });
        let prune_flip_slice = prune_table(N_FLIP, N_SLICE, ss, 18, |a, b, m| {
            (flip_mv[a][m] as usize, slice_mv[b][m] as usize)
        });
        let prune_twist_flip = prune_table(N_TWIST, N_FLIP, 0, 18, |a, b, m| {
            (twist_mv[a][m] as usize, flip_mv[b][m] as usize)
        });
        let prune_cperm_sperm = prune_table(N_CPERM, N_SPERM, 0, 10, |a, b, m| {
            (cperm_mv[a][m] as usize, sperm_mv[b][m] as usize)
        });
        let prune_udeperm_sperm = prune_table(N_UDEPERM, N_SPERM, 0, 10, |a, b, m| {
            (udeperm_mv[a][m] as usize, sperm_mv[b][m] as usize)
        });
        Tables {
            moves,
            follow,
            twist_mv,
            flip_mv,
            slice_mv,
            cperm_mv,
            udeperm_mv,
            sperm_mv,
            prune_twist_slice,
            prune_flip_slice,
            prune_twist_flip,
            prune_cperm_sperm,
            prune_udeperm_sperm,
        }
    })
}

/// Kociemba two-phase solver. Not optimal; typically finds solutions of 20 moves or
/// fewer in milliseconds.
#[derive(Clone, Debug)]
pub struct TwoPhase {
    /// Stop at the first solution with at most this many moves.
    pub target_length: u32,
}

impl Default for TwoPhase {
    fn default() -> Self {
        Self { target_length: 20 }
    }
}

impl TwoPhase {
    pub fn new(target_length: u32) -> Self {
        Self { target_length }
    }

    /// Build the move and pruning tables now rather than on the first solve.
    pub fn warm_up() {
        tables();
    }
}

struct Ctx<'a> {
    t: &'a Tables,
    start: Cube3State,
    path1: Vec<usize>,
    path2: Vec<usize>,
    best: Option<Vec<usize>>,
    /// Exclusive upper bound on the total length of solutions still worth finding.
    bound: usize,
    target: usize,
    started: Instant,
    limits: &'a SearchLimits,
    stats: SearchStats,
    stop: Option<&'static str>,
}

impl Ctx<'_> {
    fn tick(&mut self) {
        self.stats.expanded += 1;
        if let Some(max) = self.limits.max_expanded {
            if self.stats.expanded >= max {
                self.stop = Some("node limit reached");
            }
        }
        if self.stats.expanded % 4096 == 0 {
            if let Some(t) = self.limits.timeout {
                if self.started.elapsed() >= t {
                    self.stop = Some("timeout");
                }
            }
        }
    }

    fn done(&self) -> bool {
        self.stop.is_some() || self.best.as_ref().is_some_and(|b| b.len() <= self.target)
    }

    fn last_move(&self) -> Option<usize> {
        self.path2.last().or(self.path1.last()).copied()
    }

    fn phase1(&mut self, tw: usize, fl: usize, sl: usize, togo: usize) {
        if togo == 0 {
            let in_g1 = tw == 0 && fl == 0 && sl == SLICE_SOLVED as usize;
            let redundant = self.path1.last().is_some_and(|m| P2_MOVES.contains(m));
            if in_g1 && !redundant {
                self.phase2_start();
            }
            return;
        }
        let t = self.t;
        let h = t.prune_twist_slice[tw * N_SLICE + sl]
            .max(t.prune_flip_slice[fl * N_SLICE + sl])
            .max(t.prune_twist_flip[tw * N_FLIP + fl]);
        if h as usize > togo {
            return;
        }
        self.tick();
        for m in 0..18 {
            if self.done() {
                return;
            }
            if self.path1.last().is_some_and(|&p| !t.follow[p][m]) {
                continue;
            }
            self.stats.generated += 1;
            self.path1.push(m);
            self.stats.peak_stored = self.stats.peak_stored.max(self.path1.len());
            self.phase1(
                t.twist_mv[tw][m] as usize,
                t.flip_mv[fl][m] as usize,
                t.slice_mv[sl][m] as usize,
                togo - 1,
            );
            self.path1.pop();
        }
    }

    fn phase2_start(&mut self) {
        let t = self.t;
        let mut s = self.start;
        for &m in &self.path1 {
            s = s.then(&super::move_table()[m]);
        }
        let (cp, ud, sp) = (cperm(&s) as usize, udeperm(&s) as usize, sperm(&s) as usize);
        let d1 = self.path1.len();
        if self.bound <= d1 {
            return;
        }
        let h = t.prune_cperm_sperm[cp * N_SPERM + sp].max(t.prune_udeperm_sperm[ud * N_SPERM + sp])
            as usize;
        for d2 in h..self.bound - d1 {
            if self.phase2(cp, ud, sp, d2) {
                let mut sol = self.path1.clone();
                sol.extend_from_slice(&self.path2);
                self.path2.clear();
                self.bound = sol.len();
                self.best = Some(sol);
                return;
            }
            if self.stop.is_some() {
                return;
            }
        }
    }

    fn phase2(&mut self, cp: usize, ud: usize, sp: usize, togo: usize) -> bool {
        if togo == 0 {
            return cp == 0 && ud == 0 && sp == 0;
        }
        let t = self.t;
        let h =
            t.prune_cperm_sperm[cp * N_SPERM + sp].max(t.prune_udeperm_sperm[ud * N_SPERM + sp]);
        if h as usize > togo {
            return false;
        }
        self.tick();
        if self.stop.is_some() {
            return false;
        }
        for (j, &m) in P2_MOVES.iter().enumerate() {
            if self.last_move().is_some_and(|p| !t.follow[p][m]) {
                continue;
            }
            self.stats.generated += 1;
            self.path2.push(m);
            if self.phase2(
                t.cperm_mv[cp][j] as usize,
                t.udeperm_mv[ud][j] as usize,
                t.sperm_mv[sp][j] as usize,
                togo - 1,
            ) {
                return true;
            }
            self.path2.pop();
        }
        false
    }
}

impl Solver<Cube3> for TwoPhase {
    fn name(&self) -> String {
        format!("twophase(target={})", self.target_length)
    }

    fn is_optimal(&self) -> bool {
        false
    }

    fn solve(
        &self,
        _puzzle: &Cube3,
        start: &Cube3State,
        limits: &SearchLimits,
    ) -> Result<Solution<FaceTurn>, SearchError> {
        let started = Instant::now();
        let t = tables();
        let max_total = limits.max_depth.map_or(usize::MAX, |d| d as usize);
        let mut ctx = Ctx {
            t,
            start: *start,
            path1: Vec::new(),
            path2: Vec::new(),
            best: None,
            bound: max_total.saturating_add(1),
            target: (self.target_length as usize).min(max_total),
            started,
            limits,
            stats: SearchStats::default(),
            stop: None,
        };
        let (tw, fl, sl) = (
            twist(start) as usize,
            flip(start) as usize,
            slice(start) as usize,
        );
        // Some phase-1 solution has at most 12 moves (the phase-1 diameter), so a first
        // solution always exists by then; longer phase-1 prefixes can still shorten the total.
        for d1 in 0.. {
            if d1 >= ctx.bound || ctx.done() {
                break;
            }
            ctx.stats.iterations += 1;
            ctx.phase1(tw, fl, sl, d1);
        }
        ctx.stats.elapsed = started.elapsed();
        match (ctx.best, ctx.stop) {
            (Some(best), _) => Ok(Solution {
                moves: best.into_iter().map(|m| t.moves[m]).collect(),
                stats: ctx.stats,
                optimal: false,
            }),
            (None, Some(reason)) => Err(SearchError::LimitReached {
                reason,
                stats: ctx.stats,
            }),
            (None, None) if limits.max_depth.is_some() => Err(SearchError::LimitReached {
                reason: "depth limit reached",
                stats: ctx.stats,
            }),
            (None, None) => Err(SearchError::Exhausted(ctx.stats)),
        }
    }
}
