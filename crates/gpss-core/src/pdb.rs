//! Pattern databases: exact distances in an abstraction of the state space.
//!
//! A [`Coordinate`] maps every state to an index in `0..size()`, forgetting some
//! information (e.g. "only the orientations of the corners"). If the mapping is
//! compatible with every move, i.e. states with equal coordinates have successors with
//! equal coordinates, then the coordinate space is itself a puzzle (a quotient), and a
//! breadth-first search over it gives, for every coordinate value, the exact number of
//! moves needed to solve that part of the puzzle. That number is a lower bound on the
//! moves needed to solve the whole puzzle, hence an admissible heuristic (Culberson &
//! Schaeffer 1998; Korf 1997). A coordinate that forgets nothing (a perfect hash of the
//! state) yields the exact distance of every state.

use std::collections::VecDeque;

use crate::heuristic::Heuristic;
use crate::puzzle::Puzzle;

/// A move-compatible projection of a puzzle's states onto `0..size()`.
pub trait Coordinate<P: Puzzle + ?Sized> {
    /// Short name, e.g. `"corner-orientation"`.
    fn name(&self) -> &str;
    /// Number of distinct coordinate values.
    fn size(&self) -> usize;
    /// The coordinate of `state`.
    fn index(&self, state: &P::State) -> usize;
}

const UNSEEN: u8 = u8::MAX;

/// A table of exact goal distances indexed by a [`Coordinate`].
#[derive(Clone)]
pub struct PatternDatabase {
    name: String,
    table: Vec<u8>,
}

impl PatternDatabase {
    /// Build the database by breadth-first search from the goal state.
    ///
    /// The queue holds full states as representatives of their coordinate classes, which
    /// is simple and fast for small coordinate spaces (the whole 2x2x2 is 3.7M states).
    /// Very large databases (3x3x3 corners, 88M entries) should be built over coordinate
    /// move tables instead; see `docs/roadmap.md`.
    pub fn build<P: Puzzle + ?Sized, C: Coordinate<P> + ?Sized>(puzzle: &P, coord: &C) -> Self {
        let mut table = vec![UNSEEN; coord.size()];
        let mut queue = VecDeque::new();
        let start = puzzle.solved();
        table[coord.index(&start)] = 0;
        queue.push_back(start);
        while let Some(state) = queue.pop_front() {
            let d = table[coord.index(&state)];
            assert!(
                d < UNSEEN - 1,
                "pattern database {} exceeds the u8 distance range",
                coord.name()
            );
            for &mv in puzzle.moves() {
                let child = puzzle.apply(&state, mv);
                let idx = coord.index(&child);
                if table[idx] == UNSEEN {
                    table[idx] = d + 1;
                    queue.push_back(child);
                }
            }
        }
        Self {
            name: coord.name().to_string(),
            table,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Distance stored for coordinate value `index` (`u8::MAX` if unreachable).
    pub fn get(&self, index: usize) -> u8 {
        self.table[index]
    }

    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Number of entries at each distance; useful for validating against literature.
    pub fn histogram(&self) -> Vec<u64> {
        let mut h = Vec::new();
        for &d in &self.table {
            if d == UNSEEN {
                continue;
            }
            let d = d as usize;
            if h.len() <= d {
                h.resize(d + 1, 0);
            }
            h[d] += 1;
        }
        h
    }

    /// Approximate memory used by the table, in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.table.len()
    }
}

/// A heuristic that looks up a [`PatternDatabase`] through its [`Coordinate`].
pub struct PdbHeuristic<C> {
    coord: C,
    pdb: PatternDatabase,
}

impl<C> PdbHeuristic<C> {
    pub fn new<P: Puzzle + ?Sized>(puzzle: &P, coord: C) -> Self
    where
        C: Coordinate<P>,
    {
        let pdb = PatternDatabase::build(puzzle, &coord);
        Self { coord, pdb }
    }

    pub fn database(&self) -> &PatternDatabase {
        &self.pdb
    }
}

impl<P: Puzzle + ?Sized, C: Coordinate<P>> Heuristic<P> for PdbHeuristic<C> {
    fn name(&self) -> String {
        self.pdb.name().to_string()
    }
    fn estimate(&self, state: &P::State) -> u32 {
        self.pdb.get(self.coord.index(state)) as u32
    }
}
