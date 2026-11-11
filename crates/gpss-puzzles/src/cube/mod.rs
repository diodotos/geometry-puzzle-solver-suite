//! Notation shared by cube-shaped puzzles (outer-layer face turns, WCA style).

use std::fmt;
use std::str::FromStr;

/// One of the six faces. Discriminants follow the `URFDLB` order used by Kociemba.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Face {
    U = 0,
    R = 1,
    F = 2,
    D = 3,
    L = 4,
    B = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [Face::U, Face::R, Face::F, Face::D, Face::L, Face::B];

    pub fn opposite(self) -> Face {
        Face::ALL[(self as usize + 3) % 6]
    }

    pub fn letter(self) -> char {
        b"URFDLB"[self as usize] as char
    }
}

/// How far a face is turned: clockwise quarter, half, or counter-clockwise quarter
/// (as seen looking at the face).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Turn {
    Cw = 1,
    Half = 2,
    Ccw = 3,
}

impl Turn {
    pub const ALL: [Turn; 3] = [Turn::Cw, Turn::Half, Turn::Ccw];

    pub fn quarter_turns(self) -> usize {
        self as usize
    }

    pub fn inverse(self) -> Turn {
        match self {
            Turn::Cw => Turn::Ccw,
            Turn::Half => Turn::Half,
            Turn::Ccw => Turn::Cw,
        }
    }
}

/// An outer-layer face turn such as `R`, `U2` or `F'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FaceTurn {
    pub face: Face,
    pub turn: Turn,
}

impl FaceTurn {
    pub const fn new(face: Face, turn: Turn) -> Self {
        Self { face, turn }
    }

    pub fn inverse(self) -> Self {
        Self::new(self.face, self.turn.inverse())
    }

    /// All 18 face turns in `U U2 U' R R2 R' ...` order.
    pub fn all() -> impl Iterator<Item = FaceTurn> {
        Face::ALL
            .into_iter()
            .flat_map(|f| Turn::ALL.into_iter().map(move |t| FaceTurn::new(f, t)))
    }

    /// Standard face-turn pruning: never turn the same face twice in a row, and of two
    /// opposite (commuting) faces only allow the `URF` face first, so `U D` is searched
    /// but the equivalent `D U` is not.
    pub fn can_follow(prev: FaceTurn, next: FaceTurn) -> bool {
        if prev.face == next.face {
            return false;
        }
        !(prev.face.opposite() == next.face && (prev.face as u8) > (next.face as u8))
    }
}

impl fmt::Display for FaceTurn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = match self.turn {
            Turn::Cw => "",
            Turn::Half => "2",
            Turn::Ccw => "'",
        };
        write!(f, "{}{}", self.face.letter(), suffix)
    }
}

/// A token that is not a valid face turn.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid move {token:?} at position {position} (expected one of U R F D L B, optionally followed by 2 or ')")]
pub struct ParseMoveError {
    pub token: String,
    /// Zero-based index of the token in the sequence.
    pub position: usize,
}

impl FromStr for FaceTurn {
    type Err = ParseMoveError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let err = || ParseMoveError {
            token: s.to_string(),
            position: 0,
        };
        let mut chars = s.chars();
        let face = match chars.next().ok_or_else(err)? {
            'U' => Face::U,
            'R' => Face::R,
            'F' => Face::F,
            'D' => Face::D,
            'L' => Face::L,
            'B' => Face::B,
            _ => return Err(err()),
        };
        let turn = match chars.as_str() {
            "" => Turn::Cw,
            "2" => Turn::Half,
            "'" => Turn::Ccw,
            _ => return Err(err()),
        };
        Ok(FaceTurn::new(face, turn))
    }
}

/// Parse a whitespace-separated move sequence. The empty string is the empty sequence.
pub fn parse_sequence(s: &str) -> Result<Vec<FaceTurn>, ParseMoveError> {
    s.split_whitespace()
        .enumerate()
        .map(|(i, tok)| {
            tok.parse::<FaceTurn>().map_err(|mut e| {
                e.position = i;
                e
            })
        })
        .collect()
}

/// Invert a move sequence (reverse order, invert each move).
pub fn invert_sequence(moves: &[FaceTurn]) -> Vec<FaceTurn> {
    moves.iter().rev().map(|m| m.inverse()).collect()
}

/// A random WCA-style scramble: uniformly random face turns, never the same face twice
/// in a row.
pub fn random_scramble<R: rand::Rng + ?Sized>(rng: &mut R, length: usize) -> Vec<FaceTurn> {
    let mut out: Vec<FaceTurn> = Vec::with_capacity(length);
    while out.len() < length {
        let face = Face::ALL[rng.gen_range(0..6)];
        if out.last().is_some_and(|m| m.face == face) {
            continue;
        }
        out.push(FaceTurn::new(face, Turn::ALL[rng.gen_range(0..3)]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_roundtrip() {
        let s = "U U2 U' R R2 R' F F2 F' D D2 D' L L2 L' B B2 B'";
        let seq = parse_sequence(s).unwrap();
        assert_eq!(seq.len(), 18);
        assert_eq!(gpss_core::puzzle::format_moves(&seq), s);
        assert_eq!(seq, FaceTurn::all().collect::<Vec<_>>());
    }

    #[test]
    fn parse_whitespace_and_empty() {
        assert_eq!(parse_sequence("").unwrap(), vec![]);
        assert_eq!(parse_sequence("   ").unwrap(), vec![]);
        assert_eq!(parse_sequence("  R\tU'\n F2 ").unwrap().len(), 3);
    }

    #[test]
    fn parse_rejects_invalid_tokens() {
        for bad in ["X", "R3", "r", "U4", "R,U", "Rw", "R2'", "x", "R''"] {
            assert!(parse_sequence(bad).is_err(), "{bad} should be rejected");
        }
        let err = parse_sequence("R U X").unwrap_err();
        assert_eq!((err.token.as_str(), err.position), ("X", 2));
    }

    #[test]
    fn pruning_rules() {
        let m = |s: &str| s.parse::<FaceTurn>().unwrap();
        assert!(!FaceTurn::can_follow(m("R"), m("R2")));
        assert!(FaceTurn::can_follow(m("U"), m("D")));
        assert!(!FaceTurn::can_follow(m("D"), m("U")));
        assert!(FaceTurn::can_follow(m("R"), m("U")));
    }
}
