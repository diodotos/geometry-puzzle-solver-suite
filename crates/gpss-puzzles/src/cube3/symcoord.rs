//! Symmetry-reduced coordinates and pruning tables for the two-phase solver, after
//! Kociemba [KocImp].
//!
//! The 16 symmetries of `D4h` (those fixing the U-D axis, [`super::sym`]) map the
//! two-phase subgroup `G1` to itself, so conjugating a position by one of them changes
//! neither its phase-1 distance (to `G1`) nor its phase-2 distance (inside `G1`). A
//! coordinate `x` that is carried along by conjugation (`x(conj(c, s))` depends only on
//! `x(c)` and `s`) therefore splits into *symmetry classes*; a pruning table only needs
//! one row per class:
//!
//! | Coordinate | Raw size | Classes | Paired with | Table entries |
//! |---|---|---|---|---|
//! | flip x UD-slice (phase 1) | 2048 x 495 = 1,013,760 | 64,430 | twist (2,187) | 140,908,410 |
//! | corner permutation (phase 2) | 40,320 | 2,768 | U/D-edge permutation (40,320) | 111,605,760 |
//!
//! A state is looked up as `(class(x), y')`, where `s` is a symmetry taking `x` to its
//! class representative and `y'` is the partner coordinate conjugated by the same `s`.
//! When the representative is itself symmetric (its *stabiliser* is larger than the
//! identity), several `y'` describe equivalent states; the table builder writes all of
//! them so that the choice of `s` never matters.
//!
//! Tables store distances modulo 3 in two bits (Kociemba's trick): the distances of
//! neighbouring states differ by at most one, so a search that knows the exact distance
//! of a node recovers the exact distance of each child from its residue
//! ([`child_depth`]), and the exact distance of the root by walking down to the goal
//! ([`descend`]). Phase 1 thereby gets the *exact* phase-1 distance (the table covers the
//! whole phase-1 coordinate space), and phase 2 the exact distance of the corner and
//! U/D-edge permutations.
//!
//! [KocImp]: https://kociemba.org/math/imptwophase.htm

use super::sym::{self, N_SYM_D4H};
use super::Cube3State;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// A raw coordinate split into symmetry classes under the 16 `D4h` symmetries.
pub struct SymClasses {
    /// For each raw value `x`: `class << 4 | s`, where conjugating by `s` takes `x` to the
    /// representative of `class`.
    pub class_of: Vec<u32>,
    /// Raw value of each class representative (the smallest raw value in the class).
    pub rep: Vec<u32>,
    /// Bit `s` is set if symmetry `s` fixes the representative.
    pub stab: Vec<u16>,
}

impl SymClasses {
    /// Classify `0..n` given `conj(x, s)`, the raw coordinate of a state with coordinate
    /// `x` conjugated by symmetry `s` (`0..16`).
    #[allow(clippy::needless_range_loop)]
    pub fn new(n: usize, conj: impl Fn(usize, usize) -> usize) -> Self {
        const UNSET: u32 = u32::MAX;
        let inv = &sym::tables().inv;
        let mut class_of = vec![UNSET; n];
        let (mut rep, mut stab) = (Vec::new(), Vec::new());
        for x in 0..n {
            if class_of[x] != UNSET {
                continue;
            }
            let k = rep.len() as u32;
            let mut st = 0u16;
            for s in 0..N_SYM_D4H {
                let y = conj(x, s);
                if y == x {
                    st |= 1 << s;
                }
                // conj(x, s) = y, so conjugating y by inv[s] gives back x.
                if class_of[y] == UNSET {
                    class_of[y] = k << 4 | inv[s] as u32;
                }
            }
            rep.push(x as u32);
            stab.push(st);
        }
        Self {
            class_of,
            rep,
            stab,
        }
    }

    /// Number of classes.
    pub fn len(&self) -> usize {
        self.rep.len()
    }

    /// Whether there are no classes.
    pub fn is_empty(&self) -> bool {
        self.rep.is_empty()
    }

    /// `(class, symmetry)` of raw value `x`.
    #[inline]
    pub fn get(&self, x: usize) -> (usize, usize) {
        let e = self.class_of[x];
        ((e >> 4) as usize, (e & 15) as usize)
    }

    /// FNV-1a of the representatives, stored in cache files so that tables built for
    /// another class numbering are rejected.
    pub fn fingerprint(&self) -> u64 {
        let mut h = super::cache::Fnv::new();
        for r in &self.rep {
            h.update(&r.to_le_bytes());
        }
        h.0
    }
}

/// `conj[y][s]`: coordinate `y` of a state, conjugated by symmetry `s` (`0..16`).
pub fn conj_table(
    n: usize,
    set: impl Fn(&mut Cube3State, usize),
    get: impl Fn(&Cube3State) -> u16,
) -> Vec<[u16; N_SYM_D4H]> {
    let t = sym::tables();
    (0..n)
        .map(|y| {
            let mut c = Cube3State::SOLVED;
            set(&mut c, y);
            std::array::from_fn(|s| get(&t.conjugate(&c, s)))
        })
        .collect()
}

/// Distances modulo 3, four 2-bit entries per byte.
#[derive(Clone)]
pub struct Mod3Table {
    pub(crate) data: Vec<u8>,
    pub(crate) len: usize,
}

impl Mod3Table {
    /// Pack exact distances.
    pub fn from_depths(depths: &[u8]) -> Self {
        let mut data = vec![0u8; depths.len().div_ceil(4)];
        for (i, &d) in depths.iter().enumerate() {
            data[i >> 2] |= (d % 3) << ((i & 3) << 1);
        }
        Self {
            data,
            len: depths.len(),
        }
    }

    /// An all-zero table of `len` entries (filled by a cache load).
    pub(crate) fn zeroed(len: usize) -> Self {
        Self {
            data: vec![0u8; len.div_ceil(4)],
            len,
        }
    }

    /// Distance of entry `i`, modulo 3.
    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        (self.data[i >> 2] >> ((i & 3) << 1)) & 3
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the table has no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes used.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Exact distance of a child, given the exact distance `parent` of its parent and the
/// child's residue `r` (neighbours differ by at most one).
#[inline]
pub fn child_depth(parent: usize, r: u8) -> usize {
    match (r as usize + 3 - parent % 3) % 3 {
        0 => parent,
        1 => parent + 1,
        _ => parent - 1,
    }
}

/// Exact distance of `x` from the goal: repeatedly step to a neighbour whose residue is
/// one less, counting steps until `is_goal`. `residue(x)` reads the table and
/// `step(x, m)` is the neighbour reached by move `m` (`0..M`). Returns `None` as soon
/// as the distance is known to be at least `limit`.
pub fn descend<X: Copy, const M: usize>(
    mut x: X,
    limit: usize,
    is_goal: impl Fn(X) -> bool,
    residue: impl Fn(X) -> u8,
    step: impl Fn(X, usize) -> X,
) -> Option<usize> {
    let mut d = 0;
    while !is_goal(x) {
        if d + 1 >= limit {
            return None;
        }
        let want = (residue(x) + 2) % 3;
        // Read all neighbours before choosing, so the table reads overlap.
        let next: [X; M] = std::array::from_fn(|m| step(x, m));
        let res: [u8; M] = std::array::from_fn(|m| residue(next[m]));
        let m = res
            .iter()
            .position(|&r| r == want)
            .expect("pruning table is inconsistent");
        x = next[m];
        d += 1;
    }
    Some(d)
}

const UNSEEN: u8 = u8::MAX;

/// Breadth-first distances over a symmetry-reduced product coordinate
/// `class * nb + b` (all threads of the machine).
///
/// * `start`: index of the goal;
/// * `step(class, b, m)`: index of the neighbour reached by move `m` (`0..nmoves`),
///   already reduced to its class representative;
/// * `stab`, `conj_b`: stabiliser of each class and the conjugation table of the `b`
///   coordinate, used to write all equivalent entries of a self-symmetric class at once.
///
/// Returns exact distances (one byte per entry). Layers are filled forwards from the
/// frontier while it is small and backwards (an unseen entry is at depth `d + 1` if a
/// neighbour is at depth `d`) once fewer entries are unseen than are on the frontier;
/// the move set is closed under inverses, so both are exact. Writes are atomic; racing
/// threads only ever write the same value, so the result does not depend on scheduling.
pub fn sym_bfs(
    n_classes: usize,
    nb: usize,
    start: usize,
    nmoves: usize,
    stab: &[u16],
    conj_b: &[[u16; N_SYM_D4H]],
    step: impl Fn(usize, usize, usize) -> usize + Sync,
) -> Vec<u8> {
    let n = n_classes * nb;
    let table: Vec<AtomicU8> = (0..n).map(|_| AtomicU8::new(UNSEEN)).collect();
    // Mark entry `i` (and its symmetric twins) as depth `d`; returns how many were new.
    let mark = |i: usize, d: u8| -> usize {
        let (k, b) = (i / nb, i % nb);
        let mut new = 0;
        let st = stab[k];
        if st == 1 {
            return table[i]
                .compare_exchange(UNSEEN, d, Ordering::Relaxed, Ordering::Relaxed)
                .is_ok() as usize;
        }
        for (s, conj) in conj_b[b].iter().enumerate() {
            if st >> s & 1 == 1 {
                let j = k * nb + *conj as usize;
                new += table[j]
                    .compare_exchange(UNSEEN, d, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok() as usize;
            }
        }
        new
    };
    let mut seen = mark(start, 0);
    let mut frontier = seen;
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    const CHUNK: usize = 1 << 16;
    for d in 0u8.. {
        let backward = n - seen < frontier;
        let next = AtomicUsize::new(0);
        let found = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| {
                    let mut local = 0;
                    loop {
                        let lo = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if lo >= n {
                            break;
                        }
                        for i in lo..(lo + CHUNK).min(n) {
                            let v = table[i].load(Ordering::Relaxed);
                            let (k, b) = (i / nb, i % nb);
                            if backward {
                                if v == UNSEEN
                                    && (0..nmoves)
                                        .any(|m| table[step(k, b, m)].load(Ordering::Relaxed) == d)
                                {
                                    local += mark(i, d + 1);
                                }
                            } else if v == d {
                                for m in 0..nmoves {
                                    let j = step(k, b, m);
                                    if table[j].load(Ordering::Relaxed) == UNSEEN {
                                        local += mark(j, d + 1);
                                    }
                                }
                            }
                        }
                    }
                    found.fetch_add(local, Ordering::Relaxed);
                });
            }
        });
        let found = found.into_inner();
        if found == 0 {
            break;
        }
        seen += found;
        frontier = found;
    }
    assert_eq!(seen, n, "some entries are unreachable");
    table.into_iter().map(AtomicU8::into_inner).collect()
}

/// Number of entries at each distance.
pub fn histogram(depths: &[u8]) -> Vec<u64> {
    let mut h = vec![0u64; 256];
    for &d in depths {
        h[d as usize] += 1;
    }
    while h.last() == Some(&0) {
        h.pop();
    }
    h
}
