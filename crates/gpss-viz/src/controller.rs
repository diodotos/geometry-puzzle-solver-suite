//! The input state machine shared by the desktop app and the headless recorder.
//!
//! Front ends translate their native events into [`Input`] and call [`Controller::handle`];
//! everything else (gestures, camera, timeline, solver hand-off) is decided here. The
//! intended behaviour for every input, including conflicting ones, is specified in
//! `docs/viewer.md` and exercised by the scenarios under `crates/gpss-viewer/scenarios/`.

use std::fmt;

use glam::{Vec2, Vec3};
use gpss_core::Puzzle;
use gpss_puzzles::cuboid::{Cuboid, LayerTurn};
use rand::{rngs::StdRng, SeedableRng};

use crate::camera::OrbitCamera;
use crate::drag::{turn_for_drag, DragTurn};
use crate::geom::{half_extents, vec3};
use crate::pick::pick;
use crate::player::{Player, Source};
use crate::solve::{SolveError, SolveJob, Solved};

/// Pixels a press must travel before it counts as a drag rather than a click.
pub const DRAG_THRESHOLD_PX: f32 = 8.0;

/// Pointer buttons the viewer distinguishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Button {
    /// Left button / touch: turn when pressed on a sticker, orbit elsewhere.
    Primary,
    /// Right button: always orbit.
    Secondary,
}

/// One input event, already in viewport pixels (`(0, 0)` = top-left).
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
    /// Button pressed.
    PointerDown { pos: Vec2, button: Button },
    /// Pointer moved (pressed or not).
    PointerMove { pos: Vec2 },
    /// Button released.
    PointerUp { pos: Vec2 },
    /// Wheel; positive zooms in.
    Scroll { steps: f32 },
    /// Moves in the cuboid's notation, e.g. `"R U2 2F'"`.
    Moves(String),
    /// Undo the last move.
    Undo,
    /// Back to solved, nothing queued.
    Reset,
    /// Jump to a random state reached by `length` moves (deterministic for a seed).
    Scramble { length: usize, seed: u64 },
    /// Jump to the state reached from solved by these moves.
    Setup(String),
    /// Solve the state at the end of the timeline and play the solution.
    Solve,
    /// The viewport changed size.
    Resize { size: Vec2 },
}

/// What an input did. Front ends show these as status text; scenarios assert on them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Feedback {
    /// A layer turn joined the timeline.
    Turn(LayerTurn),
    /// A press and release without enough movement on a sticker: nothing happens.
    Click,
    /// A drag too close to a diagonal of the face; nothing happens.
    Ambiguous,
    /// Pending solution moves were dropped because the user took over.
    PlaybackInterrupted { dropped: usize },
    /// The last move was undone.
    Undone(LayerTurn),
    /// The input was valid but had nothing to act on.
    Ignored(&'static str),
    /// The input was refused.
    Rejected(String),
    /// Jumped to a new state.
    Scrambled(String),
    /// Back to solved.
    Reset,
    /// A solve was started (fetch it with [`Controller::take_job`]).
    Solving,
    /// The target is already solved.
    AlreadySolved,
    /// A solution was queued for playback.
    SolutionQueued { moves: String, solver: String },
    /// A solution arrived for a state the user has since changed; it was dropped.
    SolutionStale,
    /// The solver gave up.
    SolveFailed(String),
}

impl fmt::Display for Feedback {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Feedback::Turn(mv) => write!(f, "turn {mv}"),
            Feedback::Click => write!(f, "click (no drag): no move"),
            Feedback::Ambiguous => write!(f, "diagonal drag: no move"),
            Feedback::PlaybackInterrupted { dropped } => {
                write!(f, "playback interrupted ({dropped} moves dropped)")
            }
            Feedback::Undone(mv) => write!(f, "undo {mv}"),
            Feedback::Ignored(why) => write!(f, "ignored: {why}"),
            Feedback::Rejected(why) => write!(f, "rejected: {why}"),
            Feedback::Scrambled(s) => write!(f, "scrambled: {s}"),
            Feedback::Reset => write!(f, "reset"),
            Feedback::Solving => write!(f, "solving..."),
            Feedback::AlreadySolved => write!(f, "already solved"),
            Feedback::SolutionQueued { moves, solver } => {
                let n = moves.split_whitespace().count();
                write!(f, "solution ({n} moves, {solver}): {moves}")
            }
            Feedback::SolutionStale => write!(f, "solution dropped: state changed meanwhile"),
            Feedback::SolveFailed(e) => write!(f, "solve failed: {e}"),
        }
    }
}

/// The gesture in progress.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gesture {
    /// No button down.
    Idle,
    /// Orbiting the camera.
    Orbit { last: Vec2 },
    /// Pressed on a sticker; becomes a turn once the drag is long and clear enough.
    Grab {
        sticker: usize,
        start: Vec2,
        point: Vec3,
        moved: bool,
    },
    /// This press already produced a turn; the rest of it is ignored.
    Done { sticker: usize },
}

/// Puzzle, camera, gesture and solver hand-off for one viewer.
#[derive(Clone, Debug)]
pub struct Controller {
    /// Timeline and animation.
    pub player: Player,
    /// The camera.
    pub camera: OrbitCamera,
    viewport: Vec2,
    gesture: Gesture,
    hover: Option<usize>,
    next_job: u64,
    pending: Option<SolveJob>,
    job_out: Option<SolveJob>,
}

impl Controller {
    /// A solved `cuboid` in a viewport of `viewport` pixels.
    pub fn new(cuboid: Cuboid, viewport: Vec2) -> Self {
        let camera = OrbitCamera::framing(half_extents(&cuboid));
        Self {
            player: Player::new(cuboid),
            camera,
            viewport,
            gesture: Gesture::Idle,
            hover: None,
            next_job: 1,
            pending: None,
            job_out: None,
        }
    }

    /// The puzzle.
    pub fn cuboid(&self) -> &Cuboid {
        self.player.cuboid()
    }

    /// Viewport size in pixels.
    pub fn viewport(&self) -> Vec2 {
        self.viewport
    }

    /// The gesture in progress.
    pub fn gesture(&self) -> Gesture {
        self.gesture
    }

    /// The sticker to highlight: the grabbed one during a press, else the hovered one.
    pub fn highlight(&self) -> Option<usize> {
        match self.gesture {
            Gesture::Grab { sticker, .. } | Gesture::Done { sticker } => Some(sticker),
            Gesture::Orbit { .. } => None,
            Gesture::Idle => self.hover,
        }
    }

    /// Whether a solve is running.
    pub fn is_solving(&self) -> bool {
        self.pending.is_some()
    }

    /// The solve the front end should run, once, after [`Feedback::Solving`].
    pub fn take_job(&mut self) -> Option<SolveJob> {
        self.job_out.take()
    }

    /// Advance animations by `dt` seconds.
    pub fn tick(&mut self, dt: f32) -> Vec<LayerTurn> {
        self.player.tick(dt)
    }

    fn sticker_at(&self, pos: Vec2) -> Option<crate::pick::Hit> {
        let (o, d) = self.camera.ray(pos, self.viewport);
        pick(self.cuboid(), o, d)
    }

    fn user_turn(&mut self, mv: LayerTurn, out: &mut Vec<Feedback>) {
        let dropped = self.player.cancel_solver_moves();
        if dropped > 0 {
            out.push(Feedback::PlaybackInterrupted { dropped });
        }
        match self.player.push(mv, Source::User) {
            Ok(()) => out.push(Feedback::Turn(mv)),
            Err(e) => out.push(Feedback::Rejected(e.to_string())),
        }
    }

    fn jump(&mut self, moves: Vec<LayerTurn>) -> String {
        let state = self.cuboid().scrambled(&moves);
        self.player.set_state(state);
        self.pending = None;
        self.job_out = None;
        gpss_core::puzzle::format_moves(&moves)
    }

    /// Handle one input.
    pub fn handle(&mut self, input: Input) -> Vec<Feedback> {
        let mut out = Vec::new();
        match input {
            Input::PointerDown { pos, button } => {
                if self.gesture != Gesture::Idle {
                    out.push(Feedback::Ignored("a button is already down"));
                } else {
                    let hit = (button == Button::Primary)
                        .then(|| self.sticker_at(pos))
                        .flatten();
                    self.gesture = match hit {
                        Some(h) => Gesture::Grab {
                            sticker: h.sticker,
                            start: pos,
                            point: h.point,
                            moved: false,
                        },
                        None => Gesture::Orbit { last: pos },
                    };
                }
            }
            Input::PointerMove { pos } => match self.gesture {
                Gesture::Idle => self.hover = self.sticker_at(pos).map(|h| h.sticker),
                Gesture::Orbit { last } => {
                    self.camera.orbit(pos - last);
                    self.gesture = Gesture::Orbit { last: pos };
                }
                Gesture::Grab {
                    sticker,
                    start,
                    point,
                    ..
                } => {
                    if pos.distance(start) < DRAG_THRESHOLD_PX {
                        return out;
                    }
                    self.gesture = Gesture::Grab {
                        sticker,
                        start,
                        point,
                        moved: true,
                    };
                    let n = vec3(self.cuboid().stickers()[sticker].normal);
                    let (o, d) = self.camera.ray(pos, self.viewport);
                    let denom = d.dot(n);
                    if denom.abs() < 1e-4 {
                        return out;
                    }
                    let t = (point - o).dot(n) / denom;
                    if t <= 0.0 {
                        return out;
                    }
                    let drag = o + d * t - point;
                    if let DragTurn::Turn(mv) = turn_for_drag(self.cuboid(), sticker, drag) {
                        self.user_turn(mv, &mut out);
                        self.gesture = Gesture::Done { sticker };
                    }
                }
                Gesture::Done { .. } => {}
            },
            Input::PointerUp { pos } => {
                match self.gesture {
                    Gesture::Grab { moved: false, .. } => out.push(Feedback::Click),
                    Gesture::Grab { moved: true, .. } => out.push(Feedback::Ambiguous),
                    Gesture::Idle => out.push(Feedback::Ignored("no button was down")),
                    Gesture::Orbit { .. } | Gesture::Done { .. } => {}
                }
                self.gesture = Gesture::Idle;
                self.hover = self.sticker_at(pos).map(|h| h.sticker);
            }
            Input::Scroll { steps } => {
                let base = OrbitCamera::framing(half_extents(self.cuboid())).distance;
                self.camera.zoom(steps, base * 0.6, base * 3.0);
            }
            Input::Moves(text) => match self.cuboid().parse_sequence(&text) {
                Ok(moves) if moves.is_empty() => out.push(Feedback::Ignored("no moves given")),
                Ok(moves) => {
                    for mv in moves {
                        self.user_turn(mv, &mut out);
                    }
                }
                Err(e) => out.push(Feedback::Rejected(e.to_string())),
            },
            Input::Undo => {
                let dropped = self.player.cancel_solver_moves();
                if dropped > 0 {
                    out.push(Feedback::PlaybackInterrupted { dropped });
                }
                match self.player.pop() {
                    Some(step) => out.push(Feedback::Undone(step.mv)),
                    None => out.push(Feedback::Ignored("nothing to undo")),
                }
            }
            Input::Reset => {
                self.jump(Vec::new());
                out.push(Feedback::Reset);
            }
            Input::Scramble { length, seed } => {
                let mut rng = StdRng::seed_from_u64(seed);
                let moves = self.cuboid().random_scramble(&mut rng, length);
                out.push(Feedback::Scrambled(self.jump(moves)));
            }
            Input::Setup(text) => match self.cuboid().parse_sequence(&text) {
                Ok(moves) => out.push(Feedback::Scrambled(self.jump(moves))),
                Err(e) => out.push(Feedback::Rejected(e.to_string())),
            },
            Input::Solve => {
                if self.pending.is_some() {
                    out.push(Feedback::Ignored("a solve is already running"));
                } else if self.cuboid().is_solved(self.player.target()) {
                    out.push(Feedback::AlreadySolved);
                } else {
                    let job = SolveJob {
                        id: self.next_job,
                        cuboid: self.cuboid().clone(),
                        state: self.player.target().clone(),
                    };
                    self.next_job += 1;
                    self.pending = Some(job.clone());
                    self.job_out = Some(job);
                    out.push(Feedback::Solving);
                }
            }
            Input::Resize { size } => self.viewport = size.max(Vec2::ONE),
        }
        out
    }

    /// Deliver the answer to the job with this `id`.
    pub fn finish_solve(&mut self, id: u64, result: Result<Solved, SolveError>) -> Vec<Feedback> {
        if self.pending.as_ref().map(|j| j.id) != Some(id) {
            return Vec::new();
        }
        let job = self.pending.take().expect("checked above");
        if &job.state != self.player.target() {
            return vec![Feedback::SolutionStale];
        }
        match result {
            Ok(s) => {
                for &mv in &s.moves {
                    if let Err(e) = self.player.push(mv, Source::Solver) {
                        return vec![Feedback::SolveFailed(e.to_string())];
                    }
                }
                vec![Feedback::SolutionQueued {
                    moves: gpss_core::puzzle::format_moves(&s.moves),
                    solver: s.solver,
                }]
            }
            Err(e) => vec![Feedback::SolveFailed(e.to_string())],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve;

    fn ctl(dims: &str) -> Controller {
        Controller::new(Cuboid::new(dims.parse().unwrap()), Vec2::new(800.0, 600.0))
    }

    fn screen_of(c: &Controller, pos: [i32; 3], normal: [i32; 3]) -> Vec2 {
        c.camera
            .project(vec3(pos) + vec3(normal), c.viewport())
            .unwrap()
    }

    fn drag(c: &mut Controller, from: Vec2, to: Vec2) -> Vec<Feedback> {
        let mut out = c.handle(Input::PointerDown {
            pos: from,
            button: Button::Primary,
        });
        for i in 1..=10 {
            out.extend(c.handle(Input::PointerMove {
                pos: from.lerp(to, i as f32 / 10.0),
            }));
        }
        out.extend(c.handle(Input::PointerUp { pos: to }));
        out
    }

    #[test]
    fn dragging_the_right_column_of_f_upwards_is_r() {
        let mut c = ctl("3x3x3");
        let from = screen_of(&c, [2, 0, 2], [0, 0, 1]);
        let to = screen_of(&c, [2, 2, 2], [0, 0, 1]);
        let fb = drag(&mut c, from, to);
        assert_eq!(
            fb,
            vec![Feedback::Turn(c.cuboid().parse_move("R").unwrap())]
        );
    }

    #[test]
    fn one_press_makes_at_most_one_turn() {
        let mut c = ctl("3x3x3");
        let from = screen_of(&c, [2, -2, 2], [0, 0, 1]);
        let to = from + Vec2::new(0.0, -400.0);
        let turns = drag(&mut c, from, to)
            .into_iter()
            .filter(|f| matches!(f, Feedback::Turn(_)))
            .count();
        assert_eq!(turns, 1);
    }

    #[test]
    fn short_press_is_a_click_and_empty_space_orbits() {
        let mut c = ctl("3x3x3");
        let p = screen_of(&c, [0, 0, 2], [0, 0, 1]);
        assert_eq!(
            drag(&mut c, p, p + Vec2::new(3.0, 0.0)),
            vec![Feedback::Click]
        );
        let yaw = c.camera.yaw;
        assert!(drag(&mut c, Vec2::new(20.0, 20.0), Vec2::new(120.0, 20.0)).is_empty());
        assert!(c.camera.yaw != yaw);
        assert!(c.player.is_idle());
    }

    #[test]
    fn user_input_interrupts_playback_and_stale_solutions_are_dropped() {
        let mut c = ctl("3x3x3");
        c.handle(Input::Setup("R U F".into()));
        assert_eq!(c.handle(Input::Solve), vec![Feedback::Solving]);
        let job = c.take_job().unwrap();
        let solution = solve::run(&job).unwrap();
        let len = solution.moves.len();
        let fb = c.finish_solve(job.id, Ok(solution));
        assert!(matches!(fb[0], Feedback::SolutionQueued { .. }));
        c.tick(0.05);
        // The move already animating finishes; the rest of the solution is dropped.
        let fb = c.handle(Input::Moves("U".into()));
        assert_eq!(fb[0], Feedback::PlaybackInterrupted { dropped: len - 1 });
        // A solve whose state changes before the answer arrives is dropped.
        c.handle(Input::Solve);
        let job = c.take_job().unwrap();
        c.handle(Input::Moves("R".into()));
        assert_eq!(
            c.finish_solve(job.id, solve::run(&job)),
            vec![Feedback::SolutionStale]
        );
    }

    #[test]
    fn solve_then_play_back_reaches_solved() {
        let mut c = ctl("3x3x3");
        c.handle(Input::Scramble {
            length: 25,
            seed: 3,
        });
        c.handle(Input::Moves("2R U'".into()));
        c.handle(Input::Solve);
        assert_eq!(
            c.handle(Input::Solve),
            vec![Feedback::Ignored("a solve is already running")]
        );
        let job = c.take_job().unwrap();
        c.finish_solve(job.id, solve::run(&job));
        c.player.finish();
        assert!(c.cuboid().is_solved(c.player.state()));
        assert_eq!(c.handle(Input::Solve), vec![Feedback::AlreadySolved]);
    }
}
