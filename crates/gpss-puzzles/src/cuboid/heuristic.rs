//! Admissible heuristics for generic cuboids.
//!
//! [`CornerHeuristic`] projects a cuboid onto its eight corner cubies. An outer-layer
//! turn moves the corners exactly like the matching 2x2x2 face turn, and inner-layer
//! turns leave them alone. So the corners of any `A×B×C` cuboid form a 2x2x2 whose
//! moves are a subset of the 2x2x2's: half-turn-only axes only drop moves. The exact
//! 2x2x2 distance of that projection, up to whole-cube rotation, is therefore a lower
//! bound on the cuboid distance. It is the exact distance of a relaxed problem, so it
//! is also consistent. On a 2x2x2 it is the perfect heuristic. See
//! `docs/theory/cuboid.md`.

use gpss_core::pdb::PdbHeuristic;
use gpss_core::Heuristic;

use super::{Cuboid, CuboidState};
use crate::cube2::{Cube2, Cube2State, FullCoord};

/// Facelet colours of the eight corner pieces in `cube2` order
/// (`URF UFL ULB UBR DFR DLF DBL DRB`). Each corner lists its U/D facelet first, then
/// the others clockwise, which is Kociemba's convention and the one `cube2` uses.
/// Colours are face indices in `U R F D L B` order.
pub(super) const CORNER_COLOURS: [[u8; 3]; 8] = [
    [0, 1, 2],
    [0, 2, 4],
    [0, 4, 5],
    [0, 5, 1],
    [3, 2, 1],
    [3, 4, 2],
    [3, 5, 4],
    [3, 1, 5],
];

impl Cuboid {
    /// The 2x2x2 formed by this state's corner cubies, or `None` if the corner
    /// stickers do not describe eight distinct corner pieces (unreachable states only).
    pub fn corner_state(&self, state: &CuboidState) -> Option<Cube2State> {
        let mut out = Cube2State::SOLVED;
        let mut seen = [false; 8];
        for (i, facelets) in self.corners.iter().enumerate() {
            let c = facelets.map(|f| state[f as usize]);
            let ori = c.iter().position(|&x| x == 0 || x == 3)?;
            let (c1, c2) = (c[(ori + 1) % 3], c[(ori + 2) % 3]);
            let piece = CORNER_COLOURS
                .iter()
                .position(|p| p[1] == c1 && p[2] == c2)?;
            if std::mem::replace(&mut seen[piece], true) {
                return None;
            }
            out.cp[i] = piece as u8;
            out.co[i] = ori as u8;
        }
        Some(out)
    }
}

/// The full 2x2x2 distance table, generated once per process and shared by every
/// [`CornerHeuristic`].
fn corner_table() -> &'static PdbHeuristic<FullCoord> {
    static TABLE: std::sync::OnceLock<PdbHeuristic<FullCoord>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| PdbHeuristic::new(&Cube2::new(), FullCoord))
}

/// Exact 2x2x2 distance of the corner cubies (see the module docs). Admissible and
/// consistent for every cuboid. The first one built in a process generates the 3.7 MB
/// `cube2` table (about 1 s); later ones reuse it.
pub struct CornerHeuristic {
    cuboid: Cuboid,
    cube2: Cube2,
}

impl CornerHeuristic {
    /// Build the heuristic for `cuboid`.
    pub fn new(cuboid: &Cuboid) -> Self {
        corner_table();
        Self {
            cuboid: cuboid.clone(),
            cube2: Cube2::new(),
        }
    }
}

impl Heuristic<Cuboid> for CornerHeuristic {
    fn name(&self) -> String {
        "corners".into()
    }

    fn estimate(&self, state: &CuboidState) -> u32 {
        match self.cuboid.corner_state(state) {
            Some(corners) => {
                let (regripped, _) = self.cube2.regrip(&corners);
                corner_table().estimate(&regripped)
            }
            None => 0,
        }
    }
}
