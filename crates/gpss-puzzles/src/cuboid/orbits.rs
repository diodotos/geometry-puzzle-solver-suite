//! Generic sticker-orbit pattern databases for cuboids (edges, centres, wings, ...).
//!
//! Layer turns permute sticker positions, so positions split into orbits (union-find over
//! all move permutations). For a non-corner orbit and a set of colours, the *abstraction*
//! keeps only which positions of the orbit show a tracked colour (a bitmask). Because
//! moves permute stickers, a move maps abstract states to abstract states; the goal set is
//! the image of every whole-puzzle rotation of solved. The BFS distance in this abstract
//! space is therefore an admissible and consistent lower bound (a relaxed problem), and
//! [`OrbitHeuristic`] takes the max of all of them and [`CornerHeuristic`].
//!
//! Tracked colour sets are the three opposite-face pairs (U/D, R/L, F/B), merged into one
//! symbol so that tables stay small: an orbit of 24 positions with 8 tracked stickers has
//! at most C(24, 8) = 735,471 abstract states. Orbits over 64 positions, or abstractions
//! over [`MAX_STATES`], are skipped. See `docs/theory/cuboid.md`.

use std::collections::HashMap;

use gpss_core::{Heuristic, Puzzle};

use super::{CornerHeuristic, Cuboid, CuboidState};

/// Largest abstract space built per table.
pub const MAX_STATES: u64 = 3_000_000;

/// Opposite-face colour pairs, in `U R F D L B` indices.
const PAIRS: [[u8; 2]; 3] = [[0, 3], [1, 4], [2, 5]];

impl Cuboid {
    /// Sticker orbits under the layer turns, each sorted, in order of first sticker.
    pub fn sticker_orbits(&self) -> Vec<Vec<u16>> {
        let n = self.sticker_count();
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(p: &mut [usize], mut x: usize) -> usize {
            while p[x] != x {
                p[x] = p[p[x]];
                x = p[x];
            }
            x
        }
        for src in self.perms.iter().flatten() {
            for (i, &s) in src.iter().enumerate() {
                let (a, b) = (find(&mut parent, i), find(&mut parent, s as usize));
                parent[a.max(b)] = a.min(b);
            }
        }
        let mut groups: HashMap<usize, Vec<u16>> = HashMap::new();
        for i in 0..n {
            let r = find(&mut parent, i);
            groups.entry(r).or_default().push(i as u16);
        }
        let mut out: Vec<Vec<u16>> = groups.into_values().collect();
        out.sort();
        out
    }
}

fn binomial(n: u64, k: u64) -> u64 {
    (0..k.min(n - k)).fold(1u64, |acc, i| acc.saturating_mul(n - i) / (i + 1))
}

/// Exact distance table of one orbit/colour-set abstraction.
pub struct OrbitPdb {
    positions: Vec<u16>,
    tracked: [bool; 6],
    table: HashMap<u64, u8>,
}

impl OrbitPdb {
    /// Build the table, or `None` if the orbit or abstract space is too large or trivial.
    pub fn new(cuboid: &Cuboid, positions: &[u16], colours: &[u8]) -> Option<Self> {
        let mut tracked = [false; 6];
        for &c in colours {
            tracked[c as usize] = true;
        }
        let pdb = Self {
            positions: positions.to_vec(),
            tracked,
            table: HashMap::new(),
        };
        let solved = cuboid.solved();
        let ones = pdb.key(&solved).count_ones() as u64;
        let n = positions.len() as u64;
        if n > 64 || ones == 0 || ones == n || binomial(n, ones) > MAX_STATES {
            return None;
        }
        Some(pdb.build(cuboid))
    }

    fn key(&self, state: &CuboidState) -> u64 {
        self.positions
            .iter()
            .enumerate()
            .filter(|&(_, &p)| self.tracked[state[p as usize] as usize])
            .fold(0, |k, (i, _)| k | 1 << i)
    }

    fn build(mut self, cuboid: &Cuboid) -> Self {
        let mut local = vec![usize::MAX; cuboid.sticker_count()];
        for (i, &p) in self.positions.iter().enumerate() {
            local[p as usize] = i;
        }
        // new bit i = old bit perm[i]
        let perms: Vec<Vec<usize>> = cuboid
            .moves()
            .iter()
            .map(|&mv| {
                let src = cuboid.perm(mv);
                self.positions
                    .iter()
                    .map(|&p| local[src[p as usize] as usize])
                    .collect()
            })
            .collect();
        let mut frontier: Vec<u64> = (0..cuboid.rotation_count())
            .map(|k| self.key(&cuboid.rotate(&cuboid.solved(), k)))
            .collect();
        frontier.sort_unstable();
        frontier.dedup();
        for &k in &frontier {
            self.table.insert(k, 0);
        }
        let mut depth = 0u8;
        while !frontier.is_empty() {
            depth += 1;
            let mut next = Vec::new();
            for &k in &frontier {
                for perm in &perms {
                    let child = perm
                        .iter()
                        .enumerate()
                        .fold(0u64, |c, (i, &j)| c | ((k >> j) & 1) << i);
                    if let std::collections::hash_map::Entry::Vacant(e) = self.table.entry(child) {
                        e.insert(depth);
                        next.push(child);
                    }
                }
            }
            frontier = next;
        }
        self
    }

    /// Number of abstract states reached.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Always false for a built table (it holds the goal).
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Largest distance in the table.
    pub fn max_distance(&self) -> u8 {
        self.table.values().copied().max().unwrap_or(0)
    }

    /// Exact abstract distance (0 for an abstract state not in the table, which only an
    /// unreachable state can produce).
    pub fn estimate(&self, state: &CuboidState) -> u32 {
        self.table.get(&self.key(state)).copied().unwrap_or(0) as u32
    }
}

/// Max of [`CornerHeuristic`] and an [`OrbitPdb`] for every non-corner orbit and
/// opposite-face colour pair. Admissible and consistent for every cuboid.
pub struct OrbitHeuristic {
    corners: CornerHeuristic,
    pdbs: Vec<OrbitPdb>,
}

impl OrbitHeuristic {
    /// Build every table for `cuboid` (seconds on 3x3x3 / 4x4x4; not cached on disk).
    pub fn new(cuboid: &Cuboid) -> Self {
        let corner_stickers: Vec<u16> = cuboid.corners.iter().flatten().copied().collect();
        let jobs: Vec<(Vec<u16>, [u8; 2])> = cuboid
            .sticker_orbits()
            .into_iter()
            .filter(|o| !o.iter().any(|p| corner_stickers.contains(p)))
            .flat_map(|o| PAIRS.map(|pair| (o.clone(), pair)))
            .collect();
        let pdbs = std::thread::scope(|s| {
            let handles: Vec<_> = jobs
                .iter()
                .map(|(o, pair)| s.spawn(move || OrbitPdb::new(cuboid, o, pair)))
                .collect();
            handles
                .into_iter()
                .filter_map(|h| h.join().expect("orbit table build panicked"))
                .collect()
        });
        Self {
            corners: CornerHeuristic::new(cuboid),
            pdbs,
        }
    }

    /// The orbit tables (excluding the corner table).
    pub fn tables(&self) -> &[OrbitPdb] {
        &self.pdbs
    }
}

impl Heuristic<Cuboid> for OrbitHeuristic {
    fn name(&self) -> String {
        "orbits".into()
    }

    fn estimate(&self, state: &CuboidState) -> u32 {
        self.pdbs
            .iter()
            .map(|p| p.estimate(state))
            .fold(self.corners.estimate(state), u32::max)
    }
}
