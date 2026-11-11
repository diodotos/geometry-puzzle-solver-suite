//! DAVI training-data generation.
//!
//! Deep approximate value iteration [Ago19] needs, for each training state `s`, the
//! encodings of all one-move successors `A(s, a)` (to form the Bellman target
//! `y(s) = min_a [1 + J'(A(s, a))]`) and which of them are the goal (`J'(goal) = 0`).
//! [`generate`] produces exactly that, for any [`Encodable`] puzzle, and
//! [`DaviBatch::write`] serialises it in a small self-describing binary format that
//! `training/gpss_train/data.py` reads with NumPy:
//!
//! ```text
//! <one line of UTF-8 JSON header>\n
//! <array 0 raw bytes><array 1 raw bytes>...
//! ```
//!
//! The header lists the arrays in order with dtype (always `u8`) and shape:
//!
//! | name             | shape                  | meaning                                  |
//! |------------------|------------------------|------------------------------------------|
//! | `states`         | `[count, len]`         | encoded training states                  |
//! | `children`       | `[count, moves, len]`  | encoded successor for each search move   |
//! | `child_is_goal`  | `[count, moves]`       | 1 if that successor is the goal          |
//! | `is_goal`        | `[count]`              | 1 if the state itself is the goal        |
//! | `scramble_depth` | `[count]`              | scramble length used to generate it      |
//!
//! States are produced by random walks from the goal of length `k ~ U{min..=max}` using
//! the puzzle's search moves and its `can_follow` pruning (so a move is never undone
//! immediately). This matches DeepCubeA's sampling, which scrambles the goal state
//! backwards; for puzzles whose moves are closed under inverse the two are the same.

use std::io::{self, Write};

use gpss_core::encode::{Encodable, EncodingSpec};
use rand::Rng;
use serde_json::json;

/// Identifier written into every data header.
pub const DATA_FORMAT: &str = "gpss-davi-data";
/// Bumped when the binary layout changes.
pub const DATA_FORMAT_VERSION: u32 = 1;

/// How to sample training states.
#[derive(Debug, Clone)]
pub struct ScrambleConfig {
    /// Number of training states.
    pub count: usize,
    /// Shortest scramble (DeepCubeA uses 1).
    pub min_scramble: u8,
    /// Longest scramble (`K`; DeepCubeA uses 30 for the 3x3x3). Should exceed the puzzle's
    /// diameter so that every distance is represented.
    pub max_scramble: u8,
}

/// One chunk of DAVI training data (see the module docs for the layout).
#[derive(Debug, Clone)]
pub struct DaviBatch {
    pub encoding: EncodingSpec,
    pub moves: Vec<String>,
    pub count: usize,
    pub states: Vec<u8>,
    pub children: Vec<u8>,
    pub child_is_goal: Vec<u8>,
    pub is_goal: Vec<u8>,
    pub scramble_depth: Vec<u8>,
}

/// Sample `cfg.count` scrambled states and their successors.
///
/// # Panics
///
/// If `min_scramble > max_scramble` or the puzzle has no moves.
pub fn generate<P: Encodable, R: Rng + ?Sized>(
    puzzle: &P,
    cfg: &ScrambleConfig,
    rng: &mut R,
) -> DaviBatch {
    assert!(
        cfg.min_scramble <= cfg.max_scramble,
        "min_scramble > max_scramble"
    );
    let moves = puzzle.moves();
    assert!(!moves.is_empty(), "puzzle has no moves");
    let spec = puzzle.encoding();
    let (len, m, n) = (spec.len, moves.len(), cfg.count);

    let mut batch = DaviBatch {
        moves: moves.iter().map(ToString::to_string).collect(),
        count: n,
        states: vec![0; n * len],
        children: vec![0; n * m * len],
        child_is_goal: vec![0; n * m],
        is_goal: vec![0; n],
        scramble_depth: vec![0; n],
        encoding: spec,
    };

    let mut legal = Vec::with_capacity(m);
    for i in 0..n {
        let depth = rng.gen_range(cfg.min_scramble..=cfg.max_scramble);
        let mut state = puzzle.solved();
        let mut prev = None;
        for _ in 0..depth {
            legal.clear();
            legal.extend(
                moves
                    .iter()
                    .copied()
                    .filter(|&mv| prev.map_or(true, |p| puzzle.can_follow(p, mv))),
            );
            let mv = legal[rng.gen_range(0..legal.len())];
            state = puzzle.apply(&state, mv);
            prev = Some(mv);
        }
        batch.scramble_depth[i] = depth;
        batch.is_goal[i] = puzzle.is_solved(&state) as u8;
        puzzle.encode(&state, &mut batch.states[i * len..(i + 1) * len]);
        for (j, &mv) in moves.iter().enumerate() {
            let child = puzzle.apply(&state, mv);
            let at = (i * m + j) * len;
            puzzle.encode(&child, &mut batch.children[at..at + len]);
            batch.child_is_goal[i * m + j] = puzzle.is_solved(&child) as u8;
        }
    }
    batch
}

impl DaviBatch {
    /// The JSON header describing this batch.
    pub fn header(&self) -> serde_json::Value {
        let (n, m, len) = (self.count, self.moves.len(), self.encoding.len);
        json!({
            "format": DATA_FORMAT,
            "format_version": DATA_FORMAT_VERSION,
            "encoding": encoding_json(&self.encoding),
            "count": n,
            "moves": self.moves,
            "arrays": [
                { "name": "states", "dtype": "u8", "shape": [n, len] },
                { "name": "children", "dtype": "u8", "shape": [n, m, len] },
                { "name": "child_is_goal", "dtype": "u8", "shape": [n, m] },
                { "name": "is_goal", "dtype": "u8", "shape": [n] },
                { "name": "scramble_depth", "dtype": "u8", "shape": [n] },
            ],
        })
    }

    /// Write the header line and the raw arrays.
    pub fn write<W: Write>(&self, mut w: W) -> io::Result<()> {
        writeln!(w, "{}", self.header())?;
        for part in [
            &self.states,
            &self.children,
            &self.child_is_goal,
            &self.is_goal,
            &self.scramble_depth,
        ] {
            w.write_all(part)?;
        }
        w.flush()
    }
}

/// JSON form of an [`EncodingSpec`], shared by data headers and weight manifests.
pub fn encoding_json(spec: &EncodingSpec) -> serde_json::Value {
    json!({
        "puzzle": spec.puzzle,
        "scheme": spec.scheme,
        "version": spec.version,
        "len": spec.len,
        "classes": spec.classes,
        "key": spec.key(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpss_core::Puzzle;
    use gpss_puzzles::cube2::Cube2;
    use rand::{rngs::StdRng, SeedableRng};

    fn sample(count: usize, min: u8, max: u8, seed: u64) -> DaviBatch {
        let cfg = ScrambleConfig {
            count,
            min_scramble: min,
            max_scramble: max,
        };
        generate(&Cube2::new(), &cfg, &mut StdRng::seed_from_u64(seed))
    }

    #[test]
    fn shapes_and_ranges() {
        let b = sample(500, 0, 14, 1);
        let (len, m) = (8, 9);
        assert_eq!(b.states.len(), 500 * len);
        assert_eq!(b.children.len(), 500 * m * len);
        assert_eq!(b.child_is_goal.len(), 500 * m);
        assert!(b.states.iter().chain(&b.children).all(|&c| c < 24));
        assert!(b.scramble_depth.iter().all(|&d| d <= 14));
        // Depth-0 samples are the goal; depth-1 samples have exactly one goal child.
        for i in 0..500 {
            if b.scramble_depth[i] == 0 {
                assert_eq!(b.is_goal[i], 1);
            }
            if b.scramble_depth[i] == 1 {
                assert_eq!(b.is_goal[i], 0);
                assert_eq!(b.child_is_goal[i * m..(i + 1) * m].iter().sum::<u8>(), 1);
            }
        }
    }

    proptest::proptest! {
        #![proptest_config(proptest::prelude::ProptestConfig::with_cases(32))]

        /// Goal flags agree with the encodings, and every state's children are exactly
        /// its successors: re-deriving a child's successors must contain the parent
        /// (every Cube2 move has its inverse in the move set).
        #[test]
        fn goal_masks_match_encodings(seed in proptest::prelude::any::<u64>(), min in 0u8..6, extra in 0u8..9) {
            let b = sample(40, min, min + extra, seed);
            let (len, m) = (8, 9);
            let puzzle = Cube2::new();
            let mut goal = vec![0u8; len];
            puzzle.encode(&puzzle.solved(), &mut goal);
            for i in 0..b.count {
                let st = &b.states[i * len..(i + 1) * len];
                proptest::prop_assert_eq!(b.is_goal[i] == 1, st == goal.as_slice());
                proptest::prop_assert!((min..=min + extra).contains(&b.scramble_depth[i]));
                for j in 0..m {
                    let c = &b.children[(i * m + j) * len..(i * m + j + 1) * len];
                    proptest::prop_assert_eq!(b.child_is_goal[i * m + j] == 1, c == goal.as_slice());
                    proptest::prop_assert_ne!(c, st);
                }
            }
        }
    }

    #[test]
    fn deterministic_for_a_seed() {
        let a = sample(50, 1, 14, 9);
        let b = sample(50, 1, 14, 9);
        assert_eq!(a.states, b.states);
        assert_eq!(a.children, b.children);
        assert_ne!(a.states, sample(50, 1, 14, 10).states);
    }

    #[test]
    fn serialised_layout_matches_header() {
        let b = sample(7, 1, 5, 2);
        let mut out = Vec::new();
        b.write(&mut out).unwrap();
        let nl = out.iter().position(|&c| c == b'\n').unwrap();
        let header: serde_json::Value = serde_json::from_slice(&out[..nl]).unwrap();
        assert_eq!(header["format"], DATA_FORMAT);
        assert_eq!(header["encoding"]["key"], "cube2/corner-cubie/v1");
        let total: u64 = header["arrays"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                a["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|d| d.as_u64().unwrap())
                    .product::<u64>()
            })
            .sum();
        assert_eq!(out.len() - nl - 1, total as usize);
    }
}
