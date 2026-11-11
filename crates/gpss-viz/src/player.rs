//! The logical move timeline and the animation that catches the picture up with it.
//!
//! Two sequences are kept, both relative to a baseline state:
//!
//! * the **timeline**: every move the user (or the solver) has asked for, minus undone
//!   ones. Its end is the *target* state, which is what new input and solves act on.
//! * the **shown** moves: what has finished animating. `state()` is the baseline with
//!   these applied, and it is what the renderer draws.
//!
//! At each step the player compares the two. If `shown` has a move the timeline no longer
//! has (undo, a cancelled solution), it animates that move's inverse; otherwise it animates
//! the next timeline move. An animation that has started always finishes. This one rule
//! covers undo during an animation, cancelling playback and queueing input.

use gpss_core::Puzzle;
use gpss_puzzles::cuboid::{Cuboid, CuboidState, LayerTurn};

use crate::geom::{ease, in_layer, turn_rotation};

/// Who asked for a move. Solver moves can be cancelled by new user input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// Direct input: a drag or a key.
    User,
    /// Part of a solution being played back.
    Solver,
}

/// One timeline entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    /// The move.
    pub mv: LayerTurn,
    /// Who asked for it.
    pub source: Source,
}

/// The move being animated.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Animation {
    /// The rotation being shown (already inverted when reverting).
    pub mv: LayerTurn,
    /// `true` when this undoes the last shown move rather than adding one.
    pub reverting: bool,
    /// Linear progress, `0..1`.
    pub progress: f32,
}

/// Why a move was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Rejected {
    /// Not a legal move of this puzzle.
    #[error("not a legal move of this puzzle")]
    Illegal,
    /// Too many moves are already waiting to be animated.
    #[error("too many moves queued")]
    QueueFull,
}

/// At most this many timeline moves may be waiting to be animated.
pub const MAX_PENDING: usize = 64;
/// Seconds for a quarter-turn animation at normal speed; a half turn takes 1.5x.
pub const QUARTER_SECONDS: f32 = 0.3;

/// Logical state, timeline and animation of one puzzle.
#[derive(Clone, Debug)]
pub struct Player {
    cuboid: Cuboid,
    baseline: CuboidState,
    timeline: Vec<Step>,
    shown: Vec<LayerTurn>,
    state: CuboidState,
    target: CuboidState,
    anim: Option<Animation>,
    /// Multiplies animation speed (1 = normal).
    pub speed: f32,
}

impl Player {
    /// A solved puzzle with nothing queued.
    pub fn new(cuboid: Cuboid) -> Self {
        let solved = cuboid.solved();
        Self {
            cuboid,
            baseline: solved.clone(),
            timeline: Vec::new(),
            shown: Vec::new(),
            state: solved.clone(),
            target: solved,
            anim: None,
            speed: 1.0,
        }
    }

    /// The puzzle.
    pub fn cuboid(&self) -> &Cuboid {
        &self.cuboid
    }

    /// What the screen shows, ignoring the turn in progress.
    pub fn state(&self) -> &CuboidState {
        &self.state
    }

    /// Where the timeline ends: the state after every queued move.
    pub fn target(&self) -> &CuboidState {
        &self.target
    }

    /// The move being animated, if any.
    pub fn animation(&self) -> Option<Animation> {
        self.anim
    }

    /// The timeline (relative to the baseline).
    pub fn timeline(&self) -> &[Step] {
        &self.timeline
    }

    /// The moves already shown (relative to the baseline).
    pub fn shown(&self) -> &[LayerTurn] {
        &self.shown
    }

    /// Nothing animating and nothing waiting.
    pub fn is_idle(&self) -> bool {
        self.anim.is_none() && self.pending() == 0
    }

    /// Animations still needed to reach the target (not counting the one in progress).
    pub fn pending(&self) -> usize {
        // Count from the shown sequence as it will be once the current animation ends: an
        // undo may already have removed that move from the timeline.
        let mut shown = self.shown.clone();
        match self.anim {
            Some(a) if a.reverting => {
                shown.pop();
            }
            Some(a) => shown.push(a.mv),
            None => {}
        }
        let p = shown
            .iter()
            .zip(&self.timeline)
            .take_while(|(a, b)| **a == b.mv)
            .count();
        (shown.len() - p) + (self.timeline.len() - p)
    }

    /// Jump to `state` with nothing queued or animating, and make it the new baseline.
    pub fn set_state(&mut self, state: CuboidState) {
        assert_eq!(state.len(), self.cuboid.sticker_count());
        self.baseline = state.clone();
        self.state = state.clone();
        self.target = state;
        self.timeline.clear();
        self.shown.clear();
        self.anim = None;
    }

    /// Append a move to the timeline.
    pub fn push(&mut self, mv: LayerTurn, source: Source) -> Result<(), Rejected> {
        if !self.cuboid.moves().contains(&mv) {
            return Err(Rejected::Illegal);
        }
        if self.pending() >= MAX_PENDING {
            return Err(Rejected::QueueFull);
        }
        self.timeline.push(Step { mv, source });
        self.target = self.cuboid.apply(&self.target, mv);
        Ok(())
    }

    /// Remove the last timeline move (undo). It is animated back if it was already shown.
    pub fn pop(&mut self) -> Option<Step> {
        let step = self.timeline.pop()?;
        self.target = self
            .cuboid
            .apply(&self.target, self.cuboid.inverse(step.mv));
        Some(step)
    }

    /// Drop every solver move that has not started animating yet. Returns how many.
    pub fn cancel_solver_moves(&mut self) -> usize {
        let keep = self.common_prefix() + usize::from(self.anim.is_some_and(|a| !a.reverting));
        let mut dropped = 0;
        while self.timeline.len() > keep
            && self.timeline.last().map(|s| s.source) == Some(Source::Solver)
        {
            self.pop();
            dropped += 1;
        }
        dropped
    }

    fn common_prefix(&self) -> usize {
        self.shown
            .iter()
            .zip(&self.timeline)
            .take_while(|(a, b)| **a == b.mv)
            .count()
    }

    fn next_animation(&self) -> Option<Animation> {
        let p = self.common_prefix();
        if self.shown.len() > p {
            let last = *self.shown.last().expect("shown is longer than the prefix");
            Some(Animation {
                mv: self.cuboid.inverse(last),
                reverting: true,
                progress: 0.0,
            })
        } else {
            self.timeline.get(p).map(|s| Animation {
                mv: s.mv,
                reverting: false,
                progress: 0.0,
            })
        }
    }

    fn duration(&self, mv: LayerTurn) -> f32 {
        let base = if mv.turns == 2 { 1.5 } else { 1.0 } * QUARTER_SECONDS;
        // Catch up when input runs ahead of the animation.
        let backlog = (self.pending() as f32 / 2.0).clamp(1.0, 4.0);
        base / (self.speed.max(1e-3) * backlog)
    }

    /// Advance the animation by `dt` seconds. Returns the moves applied to [`Self::state`]
    /// in this step (each finished animation, in order).
    pub fn tick(&mut self, mut dt: f32) -> Vec<LayerTurn> {
        let mut done = Vec::new();
        loop {
            if self.anim.is_none() {
                self.anim = self.next_animation();
            }
            let Some(mut a) = self.anim else { break };
            let need = (1.0 - a.progress) * self.duration(a.mv);
            if dt < need {
                a.progress += dt / self.duration(a.mv);
                self.anim = Some(a);
                break;
            }
            dt -= need;
            self.state = self.cuboid.apply(&self.state, a.mv);
            if a.reverting {
                self.shown.pop();
            } else {
                self.shown.push(a.mv);
            }
            self.anim = None;
            done.push(a.mv);
        }
        done
    }

    /// Finish everything instantly (no animation).
    pub fn finish(&mut self) {
        while !self.is_idle() {
            self.tick(f32::INFINITY);
        }
    }

    /// The transient rotation of the cubie centred at `pos`.
    pub fn rotation_of(&self, pos: [i32; 3]) -> glam::Quat {
        match self.anim {
            Some(a) if in_layer(&self.cuboid, a.mv, pos) => turn_rotation(a.mv, ease(a.progress)),
            _ => glam::Quat::IDENTITY,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn cube3() -> Player {
        Player::new(Cuboid::new("3x3x3".parse().unwrap()))
    }

    fn mv(p: &Player, s: &str) -> LayerTurn {
        p.cuboid().parse_move(s).unwrap()
    }

    #[test]
    fn moves_animate_in_order_and_land_on_the_target() {
        let mut p = cube3();
        for m in ["R", "U", "F'"] {
            p.push(mv(&p, m), Source::User).unwrap();
        }
        assert_eq!(p.state(), &p.cuboid().solved());
        let first = p.tick(0.01);
        assert!(first.is_empty());
        assert_eq!(p.animation().unwrap().mv, mv(&p, "R"));
        p.finish();
        let expected = p
            .cuboid()
            .scrambled(&p.cuboid().parse_sequence("R U F'").unwrap());
        assert_eq!(p.state(), &expected);
        assert_eq!(p.target(), &expected);
    }

    #[test]
    fn undo_during_an_animation_finishes_it_then_reverts() {
        let mut p = cube3();
        p.push(mv(&p, "R"), Source::User).unwrap();
        p.tick(0.1);
        assert!(p.animation().is_some());
        p.pop();
        assert_eq!(p.target(), &p.cuboid().solved());
        let done = p.tick(1.0);
        assert_eq!(done, vec![mv(&p, "R"), mv(&p, "R'")]);
        assert!(p.is_idle());
        assert_eq!(p.state(), &p.cuboid().solved());
    }

    #[test]
    fn undo_of_a_queued_move_never_animates_it() {
        let mut p = cube3();
        p.push(mv(&p, "R"), Source::User).unwrap();
        p.push(mv(&p, "U"), Source::User).unwrap();
        p.tick(0.1);
        p.pop();
        p.finish();
        assert_eq!(p.shown(), &[mv(&p, "R")]);
    }

    #[test]
    fn cancelling_a_solution_keeps_the_turn_in_flight() {
        let mut p = cube3();
        for m in ["R", "U"] {
            p.push(mv(&p, m), Source::User).unwrap();
        }
        p.finish();
        p.push(mv(&p, "U'"), Source::Solver).unwrap();
        p.push(mv(&p, "R'"), Source::Solver).unwrap();
        p.tick(0.1);
        assert_eq!(p.cancel_solver_moves(), 1);
        p.finish();
        assert_eq!(p.shown(), &[mv(&p, "R"), mv(&p, "U"), mv(&p, "U'")]);
    }

    #[test]
    fn illegal_and_overflowing_moves_are_rejected() {
        let mut p = Player::new(Cuboid::new("3x3x2".parse().unwrap()));
        let r = gpss_puzzles::cuboid::LayerTurn {
            axis: gpss_puzzles::cuboid::Axis::X,
            layer: 0,
            turns: 1,
        };
        assert_eq!(p.push(r, Source::User), Err(Rejected::Illegal));
        let r2 = LayerTurn { turns: 2, ..r };
        for _ in 0..MAX_PENDING {
            p.push(r2, Source::User).unwrap();
        }
        assert_eq!(p.push(r2, Source::User), Err(Rejected::QueueFull));
        // The move being animated does not use up a queue slot.
        p.tick(0.01);
        assert!(p.animation().is_some());
        assert_eq!(p.pending(), MAX_PENDING - 1);
        p.push(r2, Source::User).unwrap();
        assert_eq!(p.push(r2, Source::User), Err(Rejected::QueueFull));
    }

    #[test]
    fn undoing_the_animating_move_queues_its_reversal() {
        let mut p = cube3();
        let r = mv(&p, "R");
        p.push(r, Source::User).unwrap();
        p.tick(0.01);
        assert_eq!(p.pending(), 0);
        p.pop();
        // R finishes, then is animated back.
        assert_eq!(p.pending(), 1);
        assert!(!p.is_idle());
        p.push(mv(&p, "U"), Source::User).unwrap();
        assert_eq!(p.pending(), 2);
        p.finish();
        assert_eq!(p.shown(), &[mv(&p, "U")]);
    }

    proptest! {
        /// Any interleaving of pushes, pops, cancels and ticks ends, once finished, with
        /// the shown state equal to the target, which equals the timeline applied to the
        /// baseline.
        #[test]
        fn shown_state_converges_to_the_timeline(
            ops in proptest::collection::vec((0u8..5, 0usize..64, 0.0f32..0.5), 0..60)
        ) {
            let mut p = cube3();
            let moves = p.cuboid().moves().to_vec();
            for (op, k, dt) in ops {
                match op {
                    0 | 1 => { let _ = p.push(moves[k % moves.len()], Source::User); }
                    2 => { let _ = p.push(moves[k % moves.len()], Source::Solver); }
                    3 => { p.pop(); }
                    _ => { p.cancel_solver_moves(); }
                }
                p.tick(dt);
            }
            p.finish();
            let tl: Vec<_> = p.timeline().iter().map(|s| s.mv).collect();
            prop_assert_eq!(p.shown(), &tl[..]);
            prop_assert_eq!(p.state(), p.target());
            prop_assert_eq!(p.state(), &p.cuboid().scrambled(&tl));
        }
    }
}
