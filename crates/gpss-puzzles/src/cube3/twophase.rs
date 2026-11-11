//! Kociemba's two-phase algorithm [Koc], with the symmetry-reduced pruning tables of
//! Kociemba's own implementation [KocImp].
//!
//! Phase 1 searches (IDA*) for a move sequence taking the cube into the subgroup
//! `G1 = <U, D, R2, L2, F2, B2>`, where all orientations are solved and the UD-slice
//! edges are in the slice. Phase 2 solves inside `G1` using only those ten moves. Phase 1
//! solutions are enumerated in increasing length and each is completed by an optimal
//! phase 2, so the total length keeps improving; the search stops at the first solution
//! no longer than `target_length` (or at the limits, returning the best found so far).
//!
//! Heuristics ([`super::symcoord`]):
//!
//! - **phase 1**: the exact phase-1 distance, from one table over
//!   `(flip x slice symmetry class) x twist` (64,430 x 2,187 entries, 35 MB at 2 bits);
//! - **phase 2**: the maximum of the exact distance of
//!   `(corner permutation class) x (U/D-edge permutation)` (2,768 x 40,320 entries,
//!   28 MB) and two small tables over the slice-edge permutation
//!   (corner perm x slice perm, U/D-edge perm x slice perm, 1 MB each).
//!
//! The large tables are built on first use (a few seconds, on all cores) and cached on
//! disk next to the Korf tables ([`super::cache::cache_dir`]). See
//! `docs/theory/kociemba-two-phase.md` for the derivation and timings.
//!
//! [Koc]: https://kociemba.org/math/twophase.htm
//! [KocImp]: https://kociemba.org/math/imptwophase.htm

use super::cache::{cache_dir, invalid, write_atomic, Fnv};
use super::coord::*;
use super::symcoord::{child_depth, conj_table, descend, sym_bfs, Mod3Table, SymClasses};
use super::{Cube3, Cube3State};
use crate::cube::FaceTurn;
use gpss_core::search::{SearchError, SearchLimits, SearchStats, Solution, Solver};
use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;
use std::sync::OnceLock;
use std::time::Instant;

/// Indices (in `FaceTurn::all()` order) of the phase-2 moves `U U2 U' D D2 D' R2 L2 F2 B2`.
const P2_MOVES: [usize; 10] = [0, 1, 2, 9, 10, 11, 4, 13, 7, 16];

/// Number of raw flip x slice values, indexed `slice * N_FLIP + flip`.
pub const N_FLIPSLICE: usize = N_FLIP * N_SLICE;
/// Number of flip x slice symmetry classes under the 16 `D4h` symmetries.
pub const N_FLIPSLICE_CLASSES: usize = 64_430;
/// Number of corner-permutation symmetry classes under the 16 `D4h` symmetries.
pub const N_CPERM_CLASSES: usize = 2_768;
/// Entries in the phase-1 table: flip x slice classes x twist.
pub const N_PHASE1: usize = N_FLIPSLICE_CLASSES * N_TWIST;
/// Entries in the phase-2 table: corner-permutation classes x U/D-edge permutation.
pub const N_PHASE2: usize = N_CPERM_CLASSES * N_UDEPERM;

const FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"GPSS2PHS";

/// Symmetry classes and conjugation tables: cheap (well under a second), so always
/// computed, never cached.
struct SymCoords {
    flipslice: SymClasses,
    twist_conj: Vec<[u16; 16]>,
    cperm: SymClasses,
    udeperm_conj: Vec<[u16; 16]>,
}

impl SymCoords {
    fn new() -> Self {
        let st = super::sym::tables();
        let flipslice = SymClasses::new(N_FLIPSLICE, |x, s| {
            let mut c = Cube3State::SOLVED;
            set_slice(&mut c, (x / N_FLIP) as u16);
            set_flip(&mut c, (x % N_FLIP) as u16);
            let c = st.conjugate(&c, s);
            slice(&c) as usize * N_FLIP + flip(&c) as usize
        });
        let cperm_classes = SymClasses::new(N_CPERM, |x, s| {
            let mut c = Cube3State::SOLVED;
            set_cperm(&mut c, x as u16);
            cperm(&st.conjugate(&c, s)) as usize
        });
        assert_eq!(flipslice.len(), N_FLIPSLICE_CLASSES);
        assert_eq!(cperm_classes.len(), N_CPERM_CLASSES);
        Self {
            flipslice,
            twist_conj: conj_table(N_TWIST, |c, i| set_twist(c, i as u16), twist),
            cperm: cperm_classes,
            udeperm_conj: conj_table(N_UDEPERM, |c, i| set_udeperm(c, i as u16), udeperm),
        }
    }

    /// Phase-1 table index of a state with coordinates `(twist, flip, slice)`.
    #[inline]
    fn phase1_index(&self, tw: usize, fl: usize, sl: usize) -> usize {
        let (k, s) = self.flipslice.get(sl * N_FLIP + fl);
        k * N_TWIST + self.twist_conj[tw][s] as usize
    }

    /// Phase-2 table index of a state with coordinates `(corner perm, U/D-edge perm)`.
    #[inline]
    fn phase2_index(&self, cp: usize, ud: usize) -> usize {
        let (k, s) = self.cperm.get(cp);
        k * N_UDEPERM + self.udeperm_conj[ud][s] as usize
    }
}

struct Tables {
    moves: Vec<FaceTurn>,
    follow: [[bool; 18]; 18],
    twist_mv: Vec<[u16; 18]>,
    flip_mv: Vec<[u16; 18]>,
    slice_mv: Vec<[u16; 18]>,
    cperm_mv: Vec<[u16; 10]>,
    udeperm_mv: Vec<[u16; 10]>,
    sperm_mv: Vec<[u8; 10]>,
    /// Corner permutation and slice-sorted moves under all 18 turns, tracked through
    /// phase 1 to filter phase-1 solutions cheaply.
    cperm18_mv: Vec<[u16; 18]>,
    slice_sorted_mv: Vec<[u16; 18]>,
    sym: SymCoords,
    prune_phase1: Mod3Table,
    prune_phase2: Mod3Table,
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

/// The two symmetry-reduced pruning tables: the expensive, disk-cached part.
struct BigTables {
    phase1: Mod3Table,
    phase2: Mod3Table,
}

impl BigTables {
    fn build(t: &MoveTables, sym: &SymCoords) -> Self {
        let fs = &sym.flipslice;
        let start1 = sym.phase1_index(0, 0, SLICE_SOLVED as usize);
        let phase1 = sym_bfs(
            fs.len(),
            N_TWIST,
            start1,
            18,
            &fs.stab,
            &sym.twist_conj,
            |k, tw, m| {
                let raw = fs.rep[k] as usize;
                let (sl, fl) = (raw / N_FLIP, raw % N_FLIP);
                sym.phase1_index(
                    t.twist_mv[tw][m] as usize,
                    t.flip_mv[fl][m] as usize,
                    t.slice_mv[sl][m] as usize,
                )
            },
        );
        let phase1 = Mod3Table::from_depths(&phase1);
        let cc = &sym.cperm;
        let phase2 = sym_bfs(
            cc.len(),
            N_UDEPERM,
            sym.phase2_index(0, 0),
            10,
            &cc.stab,
            &sym.udeperm_conj,
            |k, ud, j| {
                sym.phase2_index(
                    t.cperm_mv[cc.rep[k] as usize][j] as usize,
                    t.udeperm_mv[ud][j] as usize,
                )
            },
        );
        let phase2 = Mod3Table::from_depths(&phase2);
        Self { phase1, phase2 }
    }

    fn save(&self, path: &Path, sym: &SymCoords) -> io::Result<()> {
        write_atomic(path, |f| {
            f.write_all(MAGIC)?;
            f.write_all(&FORMAT_VERSION.to_le_bytes())?;
            f.write_all(&sym.flipslice.fingerprint().to_le_bytes())?;
            f.write_all(&sym.cperm.fingerprint().to_le_bytes())?;
            let mut h = Fnv::new();
            for t in [&self.phase1, &self.phase2] {
                f.write_all(&(t.len() as u64).to_le_bytes())?;
                f.write_all(&t.data)?;
                h.update(&t.data);
            }
            f.write_all(&h.0.to_le_bytes())
        })
    }

    fn load(path: &Path, sym: &SymCoords) -> io::Result<Self> {
        let mut f = io::BufReader::new(fs::File::open(path)?);
        let mut head = [0u8; 28];
        f.read_exact(&mut head)?;
        let mut expected = Vec::with_capacity(28);
        expected.extend_from_slice(MAGIC);
        expected.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
        expected.extend_from_slice(&sym.flipslice.fingerprint().to_le_bytes());
        expected.extend_from_slice(&sym.cperm.fingerprint().to_le_bytes());
        if head[..] != expected[..] {
            return Err(invalid("not a gpss two-phase table file of this version"));
        }
        let mut h = Fnv::new();
        let mut read = |len: usize| -> io::Result<Mod3Table> {
            let mut n = [0u8; 8];
            f.read_exact(&mut n)?;
            if u64::from_le_bytes(n) != len as u64 {
                return Err(invalid("unexpected table size"));
            }
            let mut t = Mod3Table::zeroed(len);
            f.read_exact(&mut t.data)?;
            h.update(&t.data);
            Ok(t)
        };
        let phase1 = read(N_PHASE1)?;
        let phase2 = read(N_PHASE2)?;
        let mut sum = [0u8; 8];
        f.read_exact(&mut sum)?;
        if u64::from_le_bytes(sum) != h.0 {
            return Err(invalid("checksum mismatch"));
        }
        Ok(Self { phase1, phase2 })
    }

    /// Load from `dir`, or build and (best effort) save there.
    fn load_or_build(dir: Option<&Path>, t: &MoveTables, sym: &SymCoords) -> Self {
        let path = dir.map(|d| d.join(format!("twophase-v{FORMAT_VERSION}.bin")));
        if let Some(p) = &path {
            if let Ok(tables) = Self::load(p, sym) {
                return tables;
            }
        }
        let tables = Self::build(t, sym);
        if let Some(p) = &path {
            let _ = tables.save(p, sym);
        }
        tables
    }
}

/// Coordinate move tables (all cheap).
struct MoveTables {
    twist_mv: Vec<[u16; 18]>,
    flip_mv: Vec<[u16; 18]>,
    slice_mv: Vec<[u16; 18]>,
    cperm_mv: Vec<[u16; 10]>,
    udeperm_mv: Vec<[u16; 10]>,
    sperm_mv: Vec<[u8; 10]>,
    cperm18_mv: Vec<[u16; 18]>,
    slice_sorted_mv: Vec<[u16; 18]>,
}

impl MoveTables {
    fn new() -> Self {
        let all: [usize; 18] = std::array::from_fn(|i| i);
        Self {
            twist_mv: move_table(N_TWIST, &all, |s, i| set_twist(s, i as u16), twist),
            flip_mv: move_table(N_FLIP, &all, |s, i| set_flip(s, i as u16), flip),
            slice_mv: move_table(N_SLICE, &all, |s, i| set_slice(s, i as u16), slice),
            cperm_mv: move_table(N_CPERM, &P2_MOVES, |s, i| set_cperm(s, i as u16), cperm),
            udeperm_mv: move_table(
                N_UDEPERM,
                &P2_MOVES,
                |s, i| set_udeperm(s, i as u16),
                udeperm,
            ),
            sperm_mv: move_table(N_SPERM, &P2_MOVES, |s, i| set_sperm(s, i as u8), sperm),
            cperm18_mv: move_table(N_CPERM, &all, |s, i| set_cperm(s, i as u16), cperm),
            slice_sorted_mv: move_table(
                N_SLICE_SORTED,
                &all,
                |s, i| set_slice_sorted(s, i as u16),
                slice_sorted,
            ),
        }
    }
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
        let mt = MoveTables::new();
        let sym = SymCoords::new();
        let big = BigTables::load_or_build(cache_dir().as_deref(), &mt, &sym);
        let MoveTables {
            twist_mv,
            flip_mv,
            slice_mv,
            cperm_mv,
            udeperm_mv,
            sperm_mv,
            cperm18_mv,
            slice_sorted_mv,
        } = mt;
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
            cperm18_mv,
            slice_sorted_mv,
            sym,
            prune_phase1: big.phase1,
            prune_phase2: big.phase2,
            prune_cperm_sperm,
            prune_udeperm_sperm,
        }
    })
}

impl Tables {
    /// Exact number of moves from `(twist, flip, slice)` to `G1`.
    fn phase1_distance(&self, tw: usize, fl: usize, sl: usize) -> usize {
        descend::<_, 18>(
            (tw, fl, sl),
            usize::MAX,
            |(tw, fl, sl)| tw == 0 && fl == 0 && sl == SLICE_SOLVED as usize,
            |(tw, fl, sl)| self.prune_phase1.get(self.sym.phase1_index(tw, fl, sl)),
            |(tw, fl, sl), m| {
                (
                    self.twist_mv[tw][m] as usize,
                    self.flip_mv[fl][m] as usize,
                    self.slice_mv[sl][m] as usize,
                )
            },
        )
        .expect("no limit")
    }

    /// Exact number of phase-2 moves solving the corner and U/D-edge permutations
    /// (ignoring the slice edges).
    fn phase2_distance(&self, cp: usize, ud: usize) -> usize {
        self.phase2_distance_below(cp, ud, usize::MAX)
            .expect("no limit")
    }

    /// [`Self::phase2_distance`] if it is below `limit`, else `None` (found sooner).
    fn phase2_distance_below(&self, cp: usize, ud: usize, limit: usize) -> Option<usize> {
        descend::<_, 10>(
            (cp, ud),
            limit,
            |(cp, ud)| cp == 0 && ud == 0,
            |(cp, ud)| self.prune_phase2.get(self.sym.phase2_index(cp, ud)),
            |(cp, ud), j| {
                (
                    self.cperm_mv[cp][j] as usize,
                    self.udeperm_mv[ud][j] as usize,
                )
            },
        )
    }
}

/// Exact phase-1 distance of `state` (moves to reach `G1`), from the symmetry-reduced
/// table. Builds the tables on first use. Exposed for tests and diagnostics.
pub fn phase1_distance(state: &Cube3State) -> usize {
    tables().phase1_distance(
        twist(state) as usize,
        flip(state) as usize,
        slice(state) as usize,
    )
}

/// Exact number of `G1` moves solving the corner and U/D-edge permutations of `state`
/// (which must be in `G1`), from the symmetry-reduced table.
pub fn phase2_corner_edge_distance(state: &Cube3State) -> usize {
    tables().phase2_distance(cperm(state) as usize, udeperm(state) as usize)
}

/// Kociemba two-phase solver. Not optimal; typically finds solutions of 20 moves or
/// fewer in milliseconds.
#[derive(Clone, Debug)]
pub struct TwoPhase {
    /// Stop at the first solution with at most this many moves.
    pub target_length: u32,
    /// Search six equivalent problems side by side (see [`TwoPhase::new`]); `false`
    /// searches only the position as given.
    pub six_directions: bool,
}

impl Default for TwoPhase {
    fn default() -> Self {
        Self::new(20)
    }
}

impl TwoPhase {
    /// A two-phase solver that stops at the first solution of at most `target_length` moves.
    ///
    /// Phase 1 always reduces to `G1` along the U-D axis, but a position may be much
    /// easier to solve along the R-L or F-B axis, or backwards. Like Kociemba's solver
    /// [KocImp], each phase-1 depth is therefore tried on six positions with the same
    /// optimal length: the cube conjugated by the 120-degree rotation about the URF-DBL
    /// diagonal (0, 1 or 2 times), and the inverses of those three. A solution of any of
    /// them translates back to a solution of the original of the same length.
    ///
    /// [KocImp]: https://kociemba.org/math/imptwophase.htm
    pub fn new(target_length: u32) -> Self {
        Self {
            target_length,
            six_directions: true,
        }
    }

    /// Build the move and pruning tables now rather than on the first solve.
    pub fn warm_up() {
        tables();
    }
}

/// One of the equivalent problems searched by [`TwoPhase`]: `start` is the original
/// position conjugated by symmetry `sym`, then inverted if `inverse`.
#[derive(Clone, Copy)]
struct Direction {
    start: Cube3State,
    sym: usize,
    inverse: bool,
    /// Phase-1 coordinates (see [`Ctx::phase1`]) and exact phase-1 distance of `start`.
    coords: [usize; 5],
    dist: usize,
}

impl Direction {
    fn new(t: &Tables, original: &Cube3State, sym: usize, inverse: bool) -> Self {
        let c = super::sym::conjugate(original, sym);
        let start = if inverse { c.inverse() } else { c };
        let coords = [
            twist(&start) as usize,
            flip(&start) as usize,
            slice(&start) as usize,
            cperm(&start) as usize,
            slice_sorted(&start) as usize,
        ];
        let dist = t.phase1_distance(coords[0], coords[1], coords[2]);
        Self {
            start,
            sym,
            inverse,
            coords,
            dist,
        }
    }

    /// Translate a solution of `self.start` (move indices) into one of the original.
    fn translate(&self, mut moves: Vec<usize>) -> Vec<usize> {
        if self.inverse {
            moves.reverse();
            for m in &mut moves {
                *m = 3 * (*m / 3) + 2 - *m % 3;
            }
        }
        let st = super::sym::tables();
        let back = st.inv[self.sym] as usize;
        for m in &mut moves {
            *m = st.move_conj[*m][back] as usize;
        }
        moves
    }
}

struct Ctx<'a> {
    t: &'a Tables,
    start: Cube3State,
    dir: Direction,
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
    /// Account for one expansion; returns false (and records why) if a limit is hit.
    fn tick(&mut self) -> bool {
        if let Some(max) = self.limits.max_expanded {
            if self.stats.expanded >= max {
                self.stop = Some("node limit reached");
                return false;
            }
        }
        if self.stats.expanded % 4096 == 0 {
            if let Some(t) = self.limits.timeout {
                if self.started.elapsed() >= t {
                    self.stop = Some("timeout");
                    return false;
                }
            }
        }
        self.stats.expanded += 1;
        true
    }

    fn done(&self) -> bool {
        self.stop.is_some() || self.best.as_ref().is_some_and(|b| b.len() <= self.target)
    }

    fn last_move(&self) -> Option<usize> {
        self.path2.last().or(self.path1.last()).copied()
    }

    /// IDA* over phase-1 coordinates `p = (twist, flip, slice, corner perm, slice
    /// sorted)` with `togo` moves left; `dist` is the exact phase-1 distance of `p`.
    fn phase1(&mut self, p: [usize; 5], dist: usize, togo: usize) {
        let [tw, fl, sl, cp, ss] = p;
        if togo == 0 {
            let in_g1 = tw == 0 && fl == 0 && sl == SLICE_SOLVED as usize;
            let redundant = self.path1.last().is_some_and(|m| P2_MOVES.contains(m));
            if in_g1 && !redundant {
                self.phase2_start(cp, ss % N_SPERM);
            }
            return;
        }
        if dist > togo {
            return;
        }
        if !self.tick() {
            return;
        }
        let t = self.t;
        // Look up all children first: the table reads are independent, so the CPU
        // overlaps their cache misses instead of taking them one recursion at a time.
        let mut children = [([0usize; 5], 0usize); 18];
        for (m, child) in children.iter_mut().enumerate() {
            let (tw2, fl2, sl2) = (
                t.twist_mv[tw][m] as usize,
                t.flip_mv[fl][m] as usize,
                t.slice_mv[sl][m] as usize,
            );
            let r = t.prune_phase1.get(t.sym.phase1_index(tw2, fl2, sl2));
            *child = ([tw2, fl2, sl2, cp, ss], child_depth(dist, r));
        }
        for (m, &(mut child, child_dist)) in children.iter().enumerate() {
            if self.done() {
                return;
            }
            if self.path1.last().is_some_and(|&p| !t.follow[p][m]) {
                continue;
            }
            // Already in G1 with fewer than five moves to go: a phase-2 move here keeps
            // the cube in G1, and no short sequence leaves G1 and comes back, so the
            // remainder would be pure phase-2 moves, which phase 2 finds anyway [KocImp].
            if dist == 0 && togo < 5 && P2_MOVES.contains(&m) {
                continue;
            }
            self.stats.generated += 1;
            if child_dist >= togo {
                continue;
            }
            self.path1.push(m);
            self.stats.peak_stored = self.stats.peak_stored.max(self.path1.len());
            child[3] = t.cperm18_mv[cp][m] as usize;
            child[4] = t.slice_sorted_mv[ss][m] as usize;
            self.phase1(child, child_dist, togo - 1);
            self.path1.pop();
        }
    }

    /// A phase-1 solution reached `G1` with corner permutation `cp` and slice
    /// permutation `sp`: filter it with cheap tables, then run phase 2.
    fn phase2_start(&mut self, cp: usize, sp: usize) {
        let t = self.t;
        let d1 = self.path1.len();
        if self.bound <= d1 {
            return;
        }
        let left = self.bound - d1;
        // Cheapest filter first: most phase-1 solutions cannot be completed within the
        // bound, and never pay for rebuilding the cube or for the exact distance.
        let h_corner = t.prune_cperm_sperm[cp * N_SPERM + sp] as usize;
        if h_corner >= left {
            return;
        }
        let mut s = self.start;
        for &m in &self.path1 {
            s = s.then(&super::move_table()[m]);
        }
        let ud = udeperm(&s) as usize;
        let h_slice = h_corner.max(t.prune_udeperm_sperm[ud * N_SPERM + sp] as usize);
        if h_slice >= left {
            return;
        }
        let Some(dce) = t.phase2_distance_below(cp, ud, left) else {
            return;
        };
        let h = h_slice.max(dce);
        for d2 in h..left {
            if self.phase2(cp, ud, sp, dce, d2) {
                let mut sol = self.path1.clone();
                sol.extend_from_slice(&self.path2);
                self.path2.clear();
                self.bound = sol.len();
                self.best = Some(self.dir.translate(sol));
                return;
            }
            if self.stop.is_some() {
                return;
            }
        }
    }

    /// `dce` is the exact distance of the corner and U/D-edge permutations `(cp, ud)`.
    fn phase2(&mut self, cp: usize, ud: usize, sp: usize, dce: usize, togo: usize) -> bool {
        if togo == 0 {
            return cp == 0 && ud == 0 && sp == 0;
        }
        let t = self.t;
        let h = (t.prune_cperm_sperm[cp * N_SPERM + sp]
            .max(t.prune_udeperm_sperm[ud * N_SPERM + sp]) as usize)
            .max(dce);
        if h > togo {
            return false;
        }
        if !self.tick() {
            return false;
        }
        for (j, &m) in P2_MOVES.iter().enumerate() {
            if self.last_move().is_some_and(|p| !t.follow[p][m]) {
                continue;
            }
            self.stats.generated += 1;
            self.path2.push(m);
            self.stats.peak_stored = self
                .stats
                .peak_stored
                .max(self.path1.len() + self.path2.len());
            let (cp2, ud2) = (t.cperm_mv[cp][j] as usize, t.udeperm_mv[ud][j] as usize);
            let r = t.prune_phase2.get(t.sym.phase2_index(cp2, ud2));
            if self.phase2(
                cp2,
                ud2,
                t.sperm_mv[sp][j] as usize,
                child_depth(dce, r),
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
        let mut dirs = vec![Direction::new(t, start, 0, false)];
        if self.six_directions {
            let (s1, st) = (super::sym::urf3(), super::sym::tables());
            let s2 = st.mult[s1][s1] as usize;
            for (sym, inverse) in [(s1, false), (s2, false), (0, true), (s1, true), (s2, true)] {
                let d = Direction::new(t, start, sym, inverse);
                // Symmetric positions give identical searches; keep one.
                if dirs.iter().all(|e| e.start != d.start) {
                    dirs.push(d);
                }
            }
        }
        let mut ctx = Ctx {
            t,
            start: *start,
            dir: dirs[0],
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
        // Some phase-1 solution has at most 12 moves (the phase-1 diameter), so a first
        // solution always exists by then; longer phase-1 prefixes can still shorten the
        // total. Depths below a direction's exact phase-1 distance cannot succeed there
        // and are skipped.
        let min_dist = dirs.iter().map(|d| d.dist).min().unwrap_or(0);
        'depths: for d1 in min_dist.. {
            for dir in &dirs {
                if d1 >= ctx.bound || ctx.done() {
                    break 'depths;
                }
                if d1 < dir.dist {
                    continue;
                }
                ctx.dir = *dir;
                ctx.start = dir.start;
                ctx.stats.iterations += 1;
                ctx.phase1(dir.coords, dir.dist, d1);
            }
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
