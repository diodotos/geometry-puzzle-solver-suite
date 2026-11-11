//! Korf's optimal 3x3x3 heuristic: the maximum of three pattern databases [Korf97].
//!
//! - **corners**: permutation and orientation of all 8 corners, 8! x 3^7 = 88,179,840
//!   entries;
//! - **edges A / edges B**: positions and orientations of 6 edges each (`UR..DF` and
//!   `DL..BR`), 12!/6! x 2^6 = 42,577,920 entries each.
//!
//! Every entry is an exact distance in a quotient of the cube, so each table and their
//! maximum are admissible, and IDA* with [`KorfHeuristic`] returns optimal solutions.
//! Distances fit in 4 bits, so the three tables take 86.7 MB packed two per byte.
//! They are built by breadth-first search over coordinate move tables (about half a
//! minute in release mode) and cached on disk; see [`cache_dir`].

use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use gpss_core::search::IdaStar;
use gpss_core::Heuristic;

use super::coord::{cperm, set_cperm, set_twist, twist, N_CPERM, N_TWIST};
use super::{move_table, Cube3, Cube3State};

/// Number of corner-database entries.
pub const N_CORNERS: usize = N_CPERM * N_TWIST;
/// Number of ordered placements of 6 distinguishable edges in 12 positions.
pub const N_EDGE_LOC: usize = 12 * 11 * 10 * 9 * 8 * 7;
/// Number of entries in each 6-edge database.
pub const N_EDGES6: usize = N_EDGE_LOC * 64;

const UNSEEN: u8 = 0xF;
const FORMAT_VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"GPSSKORF";

/// A table of 4-bit values, two per byte.
#[derive(Clone)]
pub struct NibbleTable {
    data: Vec<u8>,
    len: usize,
}

impl NibbleTable {
    /// A table of `len` entries, all set to 15.
    pub fn new(len: usize) -> Self {
        Self {
            data: vec![0xFF; len.div_ceil(2)],
            len,
        }
    }

    /// Entry `i`.
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        (self.data[i >> 1] >> ((i & 1) << 2)) & 0xF
    }

    /// Set entry `i` to `v` (`v < 16`).
    #[inline]
    pub fn set(&mut self, i: usize, v: u8) {
        let shift = (i & 1) << 2;
        let b = &mut self.data[i >> 1];
        *b = (*b & !(0xF << shift)) | (v << shift);
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of entries at each distance (entries equal to 15 are skipped).
    pub fn histogram(&self) -> Vec<u64> {
        let mut h = vec![0u64; 15];
        for i in 0..self.len {
            let v = self.get(i);
            if v != UNSEEN {
                h[v as usize] += 1;
            }
        }
        while h.last() == Some(&0) {
            h.pop();
        }
        h
    }

    /// Bytes used by the table.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Breadth-first search over `0..n` from `start`. `succ(i, out)` writes the 18 neighbours
/// of `i`. Moves are closed under inverses, so the neighbour relation is symmetric; once
/// most entries are known, layers are filled backwards (an unseen entry is at depth d+1
/// if any neighbour is at depth d), which touches far fewer entries.
fn bfs(n: usize, start: usize, succ: impl Fn(usize, &mut [usize; 18])) -> NibbleTable {
    let mut t = NibbleTable::new(n);
    t.set(start, 0);
    let mut seen = 1usize;
    let mut buf = [0usize; 18];
    for d in 0u8.. {
        assert!(d + 1 < UNSEEN, "distance exceeds 4 bits");
        let backward = seen > n / 2;
        let mut found = 0usize;
        for i in 0..n {
            let v = t.get(i);
            if backward {
                if v != UNSEEN {
                    continue;
                }
                succ(i, &mut buf);
                if buf.iter().any(|&c| t.get(c) == d) {
                    t.set(i, d + 1);
                    found += 1;
                }
            } else if v == d {
                succ(i, &mut buf);
                for &c in &buf {
                    if t.get(c) == UNSEEN {
                        t.set(c, d + 1);
                        found += 1;
                    }
                }
            }
        }
        if found == 0 {
            break;
        }
        seen += found;
    }
    t
}

/// Rank of an ordered placement of 6 distinct positions out of 12, in `0..N_EDGE_LOC`.
fn loc_rank(loc: &[u8; 6]) -> u32 {
    let mut r = 0u32;
    for k in 0..6 {
        let smaller = loc[..k].iter().filter(|&&x| x < loc[k]).count() as u32;
        r = r * (12 - k as u32) + (loc[k] as u32 - smaller);
    }
    r
}

/// Inverse of [`loc_rank`].
fn loc_unrank(mut r: u32) -> [u8; 6] {
    let mut digits = [0u32; 6];
    for k in (0..6).rev() {
        let radix = 12 - k as u32;
        digits[k] = r % radix;
        r /= radix;
    }
    let mut used = [false; 12];
    let mut loc = [0u8; 6];
    for k in 0..6 {
        let mut d = digits[k];
        for (p, u) in used.iter_mut().enumerate() {
            if !*u {
                if d == 0 {
                    *u = true;
                    loc[k] = p as u8;
                    break;
                }
                d -= 1;
            }
        }
    }
    loc
}

/// Index of the 6 edges `base..base + 6` of `s` in a 6-edge database.
fn edges6_index(s: &Cube3State, base: u8) -> usize {
    let mut loc = [0u8; 6];
    let mut ori = 0usize;
    for p in 0..12 {
        let e = s.ep[p].wrapping_sub(base);
        if e < 6 {
            loc[e as usize] = p as u8;
            ori |= (s.eo[p] as usize) << e;
        }
    }
    loc_rank(&loc) as usize * 64 + ori
}

/// Index of `s` in the corner database.
fn corners_index(s: &Cube3State) -> usize {
    cperm(s) as usize * N_TWIST + twist(s) as usize
}

/// The three Korf pattern databases.
pub struct KorfTables {
    /// Corner permutation x orientation.
    pub corners: NibbleTable,
    /// Edges `UR UF UL UB DR DF`.
    pub edges_a: NibbleTable,
    /// Edges `DL DB FR FL BL BR`.
    pub edges_b: NibbleTable,
}

impl KorfTables {
    /// Build all three tables in memory.
    pub fn build() -> Self {
        let moves = move_table();

        let cperm_mv: Vec<[u16; 18]> = (0..N_CPERM)
            .map(|i| {
                let mut s = Cube3State::SOLVED;
                set_cperm(&mut s, i as u16);
                std::array::from_fn(|m| cperm(&s.then(&moves[m])))
            })
            .collect();
        let twist_mv: Vec<[u16; 18]> = (0..N_TWIST)
            .map(|i| {
                let mut s = Cube3State::SOLVED;
                set_twist(&mut s, i as u16);
                std::array::from_fn(|m| twist(&s.then(&moves[m])))
            })
            .collect();
        let corners = bfs(N_CORNERS, corners_index(&Cube3State::SOLVED), |i, out| {
            let (p, t) = (i / N_TWIST, i % N_TWIST);
            for m in 0..18 {
                out[m] = cperm_mv[p][m] as usize * N_TWIST + twist_mv[t][m] as usize;
            }
        });

        // Where an edge at position p goes under move m, and whether it flips:
        // new.ep[q] = old.ep[t.ep[q]], new.eo[q] = old.eo[t.ep[q]] + t.eo[q].
        let mut dest = [[0u8; 12]; 18];
        let mut flips = [[0u8; 12]; 18];
        for (m, t) in moves.iter().enumerate() {
            for q in 0..12 {
                dest[m][t.ep[q] as usize] = q as u8;
                flips[m][t.ep[q] as usize] = t.eo[q];
            }
        }
        // Packed (new location rank) | (orientation xor mask << 20); N_EDGE_LOC < 2^20.
        let loc_mv: Vec<[u32; 18]> = (0..N_EDGE_LOC as u32)
            .map(|r| {
                let loc = loc_unrank(r);
                std::array::from_fn(|m| {
                    let mut nl = [0u8; 6];
                    let mut mask = 0u32;
                    for k in 0..6 {
                        nl[k] = dest[m][loc[k] as usize];
                        mask |= (flips[m][loc[k] as usize] as u32) << k;
                    }
                    loc_rank(&nl) | (mask << 20)
                })
            })
            .collect();
        let edge_succ = |i: usize, out: &mut [usize; 18]| {
            let (r, o) = (i >> 6, i & 63);
            for m in 0..18 {
                let x = loc_mv[r][m];
                out[m] = ((x & 0xF_FFFF) as usize) << 6 | (o ^ (x >> 20) as usize);
            }
        };
        let edges_a = bfs(N_EDGES6, edges6_index(&Cube3State::SOLVED, 0), edge_succ);
        let edges_b = bfs(N_EDGES6, edges6_index(&Cube3State::SOLVED, 6), edge_succ);
        Self {
            corners,
            edges_a,
            edges_b,
        }
    }

    /// Lower bound on the distance of `s` from solved.
    #[inline]
    pub fn estimate(&self, s: &Cube3State) -> u8 {
        self.corners
            .get(corners_index(s))
            .max(self.edges_a.get(edges6_index(s, 0)))
            .max(self.edges_b.get(edges6_index(s, 6)))
    }

    /// Bytes used by the three tables.
    pub fn memory_bytes(&self) -> usize {
        self.corners.memory_bytes() + self.edges_a.memory_bytes() + self.edges_b.memory_bytes()
    }

    fn tables(&self) -> [&NibbleTable; 3] {
        [&self.corners, &self.edges_a, &self.edges_b]
    }

    /// Write the tables to `path` (via a temporary file and rename).
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        let tmp = path.with_extension(format!("tmp{}-{seq}", std::process::id()));
        let write = || -> io::Result<()> {
            let mut f = io::BufWriter::new(fs::File::create(&tmp)?);
            f.write_all(MAGIC)?;
            f.write_all(&FORMAT_VERSION.to_le_bytes())?;
            let mut h = Fnv::new();
            for t in self.tables() {
                f.write_all(&(t.len as u64).to_le_bytes())?;
                f.write_all(&t.data)?;
                h.update(&t.data);
            }
            f.write_all(&h.0.to_le_bytes())?;
            f.flush()?;
            drop(f);
            fs::rename(&tmp, path)
        };
        let result = write();
        if result.is_err() {
            let _ = fs::remove_file(&tmp);
        }
        result
    }

    /// Read tables written by [`save`](Self::save), checking sizes and checksum.
    pub fn load(path: &Path) -> io::Result<Self> {
        let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, msg.to_string());
        let mut f = io::BufReader::new(fs::File::open(path)?);
        let mut head = [0u8; 12];
        f.read_exact(&mut head)?;
        if &head[..8] != MAGIC || head[8..] != FORMAT_VERSION.to_le_bytes() {
            return Err(bad("not a gpss Korf table file of this version"));
        }
        let mut h = Fnv::new();
        let mut read = |expected: usize| -> io::Result<NibbleTable> {
            let mut len = [0u8; 8];
            f.read_exact(&mut len)?;
            if u64::from_le_bytes(len) != expected as u64 {
                return Err(bad("unexpected table size"));
            }
            let mut t = NibbleTable::new(expected);
            f.read_exact(&mut t.data)?;
            h.update(&t.data);
            Ok(t)
        };
        let corners = read(N_CORNERS)?;
        let edges_a = read(N_EDGES6)?;
        let edges_b = read(N_EDGES6)?;
        let mut sum = [0u8; 8];
        f.read_exact(&mut sum)?;
        if u64::from_le_bytes(sum) != h.0 {
            return Err(bad("checksum mismatch"));
        }
        Ok(Self {
            corners,
            edges_a,
            edges_b,
        })
    }

    /// Load from `dir` if a valid cache file is there; otherwise build, and try to save.
    /// A cache that cannot be written is not an error: the tables are still returned.
    pub fn load_or_build(dir: Option<&Path>) -> Self {
        let path = dir.map(|d| d.join(format!("korf-v{FORMAT_VERSION}.bin")));
        if let Some(p) = &path {
            if let Ok(t) = Self::load(p) {
                return t;
            }
        }
        let t = Self::build();
        if let Some(p) = &path {
            let _ = t.save(p);
        }
        t
    }
}

/// 64-bit FNV-1a, enough to detect truncated or corrupted cache files.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// Directory for cached tables: `$GPSS_CACHE_DIR`, else `$XDG_CACHE_HOME/gpss`, else
/// `$HOME/.cache/gpss`. `GPSS_CACHE_DIR=""` disables caching.
pub fn cache_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("GPSS_CACHE_DIR") {
        return (!d.is_empty()).then(|| PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("XDG_CACHE_HOME").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d).join("gpss"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache").join("gpss"))
}

/// The process-wide tables, loaded from [`cache_dir`] or built on first use.
pub fn tables() -> &'static KorfTables {
    static T: OnceLock<KorfTables> = OnceLock::new();
    T.get_or_init(|| KorfTables::load_or_build(cache_dir().as_deref()))
}

/// Korf's heuristic as a [`Heuristic`] for the generic searches.
#[derive(Clone, Copy)]
pub struct KorfHeuristic {
    tables: &'static KorfTables,
}

impl KorfHeuristic {
    /// Uses the process-wide [`tables`], building or loading them if needed.
    pub fn new() -> Self {
        Self { tables: tables() }
    }
}

impl Default for KorfHeuristic {
    fn default() -> Self {
        Self::new()
    }
}

impl Heuristic<Cube3> for KorfHeuristic {
    fn name(&self) -> String {
        "korf".into()
    }
    fn estimate(&self, state: &Cube3State) -> u32 {
        self.tables.estimate(state) as u32
    }
}

/// Korf's optimal solver: IDA* with [`KorfHeuristic`].
pub fn solver() -> IdaStar<KorfHeuristic> {
    IdaStar::new(KorfHeuristic::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loc_rank_round_trips() {
        for r in (0..N_EDGE_LOC as u32).step_by(997) {
            assert_eq!(loc_rank(&loc_unrank(r)), r);
        }
        assert_eq!(loc_rank(&[0, 1, 2, 3, 4, 5]), 0);
        assert_eq!(loc_rank(&[11, 10, 9, 8, 7, 6]), N_EDGE_LOC as u32 - 1);
    }

    #[test]
    fn nibbles() {
        let mut t = NibbleTable::new(5);
        t.set(3, 7);
        t.set(4, 2);
        assert_eq!((t.get(2), t.get(3), t.get(4)), (15, 7, 2));
    }
}
