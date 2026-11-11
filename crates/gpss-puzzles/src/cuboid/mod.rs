//! Generic `A×B×C` twisty cuboids ("Rubik's cuboids") at the sticker level.
//!
//! One model covers every cube and cuboid with at least two layers on each axis:
//! 2x2x2, 3x3x3, 4x4x4, ..., the 3x3x2 "domino", 2x2x3, 2x3x4, and so on. It is
//! deliberately simple and generic. The specialised [`crate::cube2`] and [`crate::cube3`]
//! models are much faster for the puzzles they cover. Theory and conventions are in
//! `docs/theory/cuboid.md`.
//!
//! * **Dimensions.** `A×B×C` means `A` layers on the R–L axis, `B` on the F–B axis and
//!   `C` on the U–D axis. The last number is the height, as in common usage
//!   (`3x3x2` is the domino).
//! * **Moves.** A [`LayerTurn`] turns a single layer, counted from the R, U or F face
//!   (layer 0 is the outer layer). Quarter turns are only legal on an axis whose two
//!   perpendicular dimensions are equal; otherwise that axis allows half turns only.
//! * **Notation.** A face letter turns that face's outer layer. A numeric prefix `kX`
//!   turns only the `k`-th layer from face `X` (SiGN single-slice notation), so `2R` on
//!   a 3x3x3 is the middle slice turned like `R`. Search output only uses `R`, `U` and
//!   `F` based tokens; input may use all six faces.
//! * **Goal and metric.** Every layer turn costs 1. A state is solved when every face
//!   shows a single colour, so the whole-puzzle orientation is free. Search uses all
//!   layers, which keeps distances exact in this metric. For a 2x2x2 that is half-turn
//!   metric; for a 3x3x3 it is the slice-turn metric.

mod heuristic;
#[cfg(test)]
mod tests;

pub use heuristic::CornerHeuristic;

use std::fmt;
use std::str::FromStr;

use gpss_core::search::{SearchError, SearchLimits, Solution, Solver};
use gpss_core::Puzzle;
use rand::Rng;

/// A rotation axis, named by its positive face: `X` = R, `Y` = U, `Z` = F.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Axis {
    /// The R–L axis.
    X = 0,
    /// The U–D axis.
    Y = 1,
    /// The F–B axis.
    Z = 2,
}

impl Axis {
    const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];

    fn positive_face(self) -> char {
        ['R', 'U', 'F'][self as usize]
    }

    fn unit(self) -> [i32; 3] {
        let mut v = [0; 3];
        v[self as usize] = 1;
        v
    }
}

/// Layer counts per axis, indexed by [`Axis`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Dims(pub [u8; 3]);

impl Dims {
    /// Dimensions from the `A×B×C` name order: R–L width, F–B depth, U–D height.
    pub fn new(width: u8, depth: u8, height: u8) -> Self {
        Dims([width, height, depth])
    }

    /// Layers on `axis`.
    pub fn layers(self, axis: Axis) -> u8 {
        self.0[axis as usize]
    }

    /// Whether quarter turns are legal about `axis`.
    pub fn allows_quarter(self, axis: Axis) -> bool {
        let others: Vec<u8> = Axis::ALL
            .iter()
            .filter(|&&a| a != axis)
            .map(|&a| self.layers(a))
            .collect();
        others[0] == others[1]
    }
}

impl fmt::Display for Dims {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [x, y, z] = self.0;
        write!(f, "{x}x{z}x{y}")
    }
}

/// The dimension string was not of the form `AxBxC` with every value in `2..=MAX_LAYERS`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid cuboid dimensions {0:?} (expected AxBxC with each value in 2..={MAX_LAYERS})")]
pub struct ParseDimsError(pub String);

/// Largest supported layer count per axis (sticker indices must fit in a `u16`).
pub const MAX_LAYERS: u8 = 32;

impl FromStr for Dims {
    type Err = ParseDimsError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || ParseDimsError(s.to_string());
        let parts: Vec<u8> = s
            .split('x')
            .map(|p| p.parse::<u8>().map_err(|_| err()))
            .collect::<Result<_, _>>()?;
        match parts[..] {
            [a, b, c] if [a, b, c].iter().all(|v| (2..=MAX_LAYERS).contains(v)) => {
                Ok(Dims::new(a, b, c))
            }
            _ => Err(err()),
        }
    }
}

/// Turn of a single layer: `layer` counts from the positive face of `axis` (0 = outer),
/// `turns` is the number of clockwise quarter turns seen from that face (1, 2 or 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LayerTurn {
    /// Rotation axis.
    pub axis: Axis,
    /// Layer index from the positive face.
    pub layer: u8,
    /// Clockwise quarter turns (1, 2 or 3).
    pub turns: u8,
}

impl fmt::Display for LayerTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.layer > 0 {
            write!(f, "{}", self.layer + 1)?;
        }
        let suffix = ["", "2", "'"][self.turns as usize - 1];
        write!(f, "{}{suffix}", self.axis.positive_face())
    }
}

/// A move token that this cuboid cannot perform.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseLayerTurnError {
    /// Not of the form `[k]X[2|']` with `X` in `URFDLB`.
    #[error("invalid move token {0:?}")]
    Syntax(String),
    /// The layer prefix exceeds the number of layers on that axis.
    #[error("move {0:?}: layer out of range for this cuboid")]
    LayerOutOfRange(String),
    /// A quarter turn on an axis whose cross-section is not square.
    #[error("move {0:?}: only half turns are possible about this axis")]
    QuarterNotAllowed(String),
}

#[derive(Clone, Copy)]
struct Sticker {
    pos: [i32; 3],
    normal: [i32; 3],
}

/// Rotate `v` by -90 degrees about the unit vector `axis` (clockwise seen from `+axis`).
fn rotate_cw(v: [i32; 3], axis: [i32; 3]) -> [i32; 3] {
    let c = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    let d = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    [
        -c[0] + axis[0] * d,
        -c[1] + axis[1] * d,
        -c[2] + axis[2] * d,
    ]
}

/// Sticker colours (face indices in `U R F D L B` order), one byte per sticker.
pub type CuboidState = Box<[u8]>;

/// An `A×B×C` cuboid with precomputed sticker permutations for every legal layer turn.
#[derive(Clone, Debug)]
pub struct Cuboid {
    dims: Dims,
    name: String,
    solved: CuboidState,
    /// Sticker index ranges of the six faces.
    faces: Vec<std::ops::Range<usize>>,
    moves: Vec<LayerTurn>,
    /// `perms[slot(mv)]` is `src` with `new[i] = old[src[i]]`.
    perms: Vec<Option<Box<[u16]>>>,
    layer_offset: [usize; 3],
    /// Sticker indices of each corner position's facelets, in `cube2` corner order.
    corners: [[u16; 3]; 8],
}

const FACES: [(char, Axis, i32); 6] = [
    ('U', Axis::Y, 1),
    ('R', Axis::X, 1),
    ('F', Axis::Z, 1),
    ('D', Axis::Y, -1),
    ('L', Axis::X, -1),
    ('B', Axis::Z, -1),
];

impl Cuboid {
    /// Build the cuboid with the given dimensions.
    ///
    /// # Panics
    /// If any dimension is outside `2..=MAX_LAYERS`.
    pub fn new(dims: Dims) -> Self {
        assert!(
            dims.0.iter().all(|d| (2..=MAX_LAYERS).contains(d)),
            "cuboid dimensions must be in 2..={MAX_LAYERS}"
        );
        let extent = |a: Axis| dims.layers(a) as i32 - 1;
        let mut stickers = Vec::new();
        let mut faces = Vec::new();
        let mut solved = Vec::new();
        for (colour, &(_, axis, sign)) in FACES.iter().enumerate() {
            let start = stickers.len();
            let coords = |a: Axis| (0..dims.layers(a) as i32).map(move |i| 2 * i - extent(a));
            for x in coords(Axis::X) {
                for y in coords(Axis::Y) {
                    for z in coords(Axis::Z) {
                        let pos = [x, y, z];
                        if pos[axis as usize] == sign * extent(axis) {
                            let mut normal = [0; 3];
                            normal[axis as usize] = sign;
                            stickers.push(Sticker { pos, normal });
                            solved.push(colour as u8);
                        }
                    }
                }
            }
            faces.push(start..stickers.len());
        }
        let index: std::collections::HashMap<([i32; 3], [i32; 3]), usize> = stickers
            .iter()
            .enumerate()
            .map(|(i, s)| ((s.pos, s.normal), i))
            .collect();

        let mut layer_offset = [0; 3];
        let mut total_layers = 0;
        for a in Axis::ALL {
            layer_offset[a as usize] = total_layers;
            total_layers += dims.layers(a) as usize;
        }
        let mut perms = vec![None; total_layers * 3];
        let mut moves = Vec::new();
        for axis in Axis::ALL {
            let quarter_ok = dims.allows_quarter(axis);
            for layer in 0..dims.layers(axis) {
                let level = extent(axis) - 2 * layer as i32;
                for turns in 1..=3u8 {
                    if !quarter_ok && turns != 2 {
                        continue;
                    }
                    let mut src: Vec<u16> = (0..stickers.len() as u16).collect();
                    for (i, s) in stickers.iter().enumerate() {
                        if s.pos[axis as usize] != level {
                            continue;
                        }
                        let (mut p, mut n) = (s.pos, s.normal);
                        for _ in 0..turns {
                            p = rotate_cw(p, axis.unit());
                            n = rotate_cw(n, axis.unit());
                        }
                        src[index[&(p, n)]] = i as u16;
                    }
                    let mv = LayerTurn { axis, layer, turns };
                    perms
                        [(layer_offset[axis as usize] + layer as usize) * 3 + turns as usize - 1] =
                        Some(src.into_boxed_slice());
                    moves.push(mv);
                }
            }
        }
        let corners = heuristic::CORNER_COLOURS.map(|faces| {
            let normals = faces.map(|f| {
                let (_, axis, sign) = FACES[f as usize];
                let mut n = [0; 3];
                n[axis as usize] = sign;
                n
            });
            let mut pos = [0; 3];
            for n in normals {
                for a in Axis::ALL {
                    pos[a as usize] += n[a as usize] * extent(a);
                }
            }
            normals.map(|n| index[&(pos, n)] as u16)
        });
        Cuboid {
            dims,
            name: dims.to_string(),
            solved: solved.into_boxed_slice(),
            faces,
            moves,
            perms,
            layer_offset,
            corners,
        }
    }

    /// The dimensions.
    pub fn dims(&self) -> Dims {
        self.dims
    }

    /// Number of stickers in a state.
    pub fn sticker_count(&self) -> usize {
        self.solved.len()
    }

    fn perm(&self, mv: LayerTurn) -> &[u16] {
        let slot =
            (self.layer_offset[mv.axis as usize] + mv.layer as usize) * 3 + mv.turns as usize - 1;
        self.perms[slot]
            .as_deref()
            .unwrap_or_else(|| panic!("move {mv} is not legal on a {} cuboid", self.name))
    }

    /// Parse one move token (see the module docs for the notation).
    pub fn parse_move(&self, token: &str) -> Result<LayerTurn, ParseLayerTurnError> {
        let syntax = || ParseLayerTurnError::Syntax(token.to_string());
        let digits = token
            .find(|c: char| !c.is_ascii_digit())
            .ok_or_else(syntax)?;
        let k: usize = if digits == 0 {
            1
        } else {
            token[..digits].parse().map_err(|_| syntax())?
        };
        let mut rest = token[digits..].chars();
        let letter = rest.next().ok_or_else(syntax)?;
        let turns = match rest.as_str() {
            "" => 1,
            "2" => 2,
            "'" => 3,
            _ => return Err(syntax()),
        };
        let &(_, axis, sign) = FACES.iter().find(|f| f.0 == letter).ok_or_else(syntax)?;
        let layers = self.dims.layers(axis) as usize;
        if k == 0 || k > layers {
            return Err(ParseLayerTurnError::LayerOutOfRange(token.to_string()));
        }
        if turns != 2 && !self.dims.allows_quarter(axis) {
            return Err(ParseLayerTurnError::QuarterNotAllowed(token.to_string()));
        }
        Ok(if sign > 0 {
            LayerTurn {
                axis,
                layer: (k - 1) as u8,
                turns,
            }
        } else {
            LayerTurn {
                axis,
                layer: (layers - k) as u8,
                turns: 4 - turns,
            }
        })
    }

    /// Parse a whitespace-separated move sequence.
    pub fn parse_sequence(&self, s: &str) -> Result<Vec<LayerTurn>, ParseLayerTurnError> {
        s.split_whitespace().map(|t| self.parse_move(t)).collect()
    }

    /// A random sequence of `length` search moves with no prunable adjacent pair.
    pub fn random_scramble<R: Rng + ?Sized>(&self, rng: &mut R, length: usize) -> Vec<LayerTurn> {
        let mut out: Vec<LayerTurn> = Vec::with_capacity(length);
        while out.len() < length {
            let mv = self.moves[rng.gen_range(0..self.moves.len())];
            if out.last().map_or(true, |&p| self.can_follow(p, mv)) {
                out.push(mv);
            }
        }
        out
    }

    /// The state reached from solved by `scramble`.
    pub fn scrambled(&self, scramble: &[LayerTurn]) -> CuboidState {
        self.apply_all(&self.solved, scramble)
    }

    /// Solve the position reached from solved by `scramble`.
    pub fn solve_scramble<S: Solver<Cuboid> + ?Sized>(
        &self,
        scramble: &[LayerTurn],
        solver: &S,
        limits: &SearchLimits,
    ) -> Result<Solution<LayerTurn>, SearchError> {
        solver.solve(self, &self.scrambled(scramble), limits)
    }
}

impl Puzzle for Cuboid {
    type State = CuboidState;
    type Move = LayerTurn;

    fn name(&self) -> &str {
        &self.name
    }

    fn solved(&self) -> CuboidState {
        self.solved.clone()
    }

    /// Every face shows a single colour; the orientation of the whole puzzle is free.
    fn is_solved(&self, state: &CuboidState) -> bool {
        self.faces
            .iter()
            .all(|r| state[r.clone()].iter().all(|&c| c == state[r.start]))
    }

    fn moves(&self) -> &[LayerTurn] {
        &self.moves
    }

    fn apply(&self, state: &CuboidState, mv: LayerTurn) -> CuboidState {
        self.perm(mv).iter().map(|&i| state[i as usize]).collect()
    }

    fn inverse(&self, mv: LayerTurn) -> LayerTurn {
        LayerTurn {
            turns: 4 - mv.turns,
            ..mv
        }
    }

    /// Layers on one axis commute, so turns on the same axis must strictly increase
    /// in layer index (this also forbids turning the same layer twice in a row).
    fn can_follow(&self, prev: LayerTurn, next: LayerTurn) -> bool {
        prev.axis != next.axis || next.layer > prev.layer
    }
}
