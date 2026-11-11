//! Scenario scripts for the headless recorder.
//!
//! One command per line; `#` starts a comment. Times are in seconds.
//!
//! ```text
//! cuboid 3x3x3            # puzzle (first command; default 3x3x3)
//! size 640 480            # frame size (default 640x480)
//! case <title>            # start a test case: shown as a caption, groups expectations
//! note <text>             # second caption line (the intended behaviour)
//! wait <s>                # let time pass
//! settle                  # wait until nothing is animating or queued
//! press <point> [right]   # button down (primary unless `right`)
//! move <point> <s>        # move the pointer there linearly over <s>
//! release                 # button up where the pointer is
//! drag <point> <point> <s> [right]   # press, move, release
//! scroll <steps>
//! moves <notation> | undo | reset | solve | scramble <len> <seed> | setup <notation>
//! home                    # put the camera back to its starting view
//! solve-latency <s>       # delay before a solve's answer is delivered (default 0)
//! expect solved | expect unsolved | expect state <notation> | expect idle
//! expect feedback <text>  # some feedback in this case contains <text>
//! expect no-feedback <text>
//! expect camera-moved | expect camera-still   # compared with the start of the case
//! ```
//!
//! A `<point>` is `x,y` (pixels), `F@2,0,2` (centre of the sticker facing F on the cubie
//! at doubled coordinates `2,0,2`, projected with the current camera), optionally followed
//! by `+dx,dy` pixels (the sticker must be the one visible at its centre); or `+dx,dy` alone, relative to where the current press started.

use std::fmt;

/// Where the pointer goes.
#[derive(Clone, Debug, PartialEq)]
pub enum Point {
    /// Absolute pixels.
    Pixel(f32, f32),
    /// A sticker centre plus a pixel offset.
    Sticker {
        /// Face letter (`U R F D L B`).
        face: char,
        /// Cubie centre in doubled coordinates.
        pos: [i32; 3],
        /// Pixel offset.
        offset: (f32, f32),
    },
    /// Offset from where the current press started.
    FromPress(f32, f32),
}

/// What an `expect` checks.
#[derive(Clone, Debug, PartialEq)]
pub enum Expect {
    /// The backend state is solved.
    Solved,
    /// The backend state is not solved.
    Unsolved,
    /// The backend state equals solved + these moves.
    State(String),
    /// Nothing animating or queued.
    Idle,
    /// Some feedback in the current case contains this text.
    Feedback(String),
    /// No feedback in the current case contains this text.
    NoFeedback(String),
    /// The camera changed since the case started.
    CameraMoved,
    /// The camera did not change since the case started.
    CameraStill,
}

/// One scenario command.
#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    /// Puzzle dimensions, e.g. `3x3x2`.
    Cuboid(String),
    /// Frame size.
    Size(u32, u32),
    /// Start a case.
    Case(String),
    /// Second caption line.
    Note(String),
    /// Let time pass.
    Wait(f32),
    /// Wait until idle.
    Settle,
    /// Button down.
    Press(Point, bool),
    /// Pointer move over a duration.
    Move(Point, f32),
    /// Button up.
    Release,
    /// Wheel.
    Scroll(f32),
    /// Moves in notation.
    Moves(String),
    /// Undo.
    Undo,
    /// Reset.
    Reset,
    /// Solve.
    Solve,
    /// Random scramble.
    Scramble(usize, u64),
    /// Jump to solved + moves.
    Setup(String),
    /// Camera back to the starting view.
    Home,
    /// Solver answer delay.
    SolveLatency(f32),
    /// Assertion.
    Expect(Expect),
}

/// A parse error with its line number.
#[derive(Debug, thiserror::Error)]
#[error("line {line}: {msg}")]
pub struct ParseError {
    /// 1-based line.
    pub line: usize,
    /// What was wrong.
    pub msg: String,
}

fn pair(s: &str) -> Option<(f32, f32)> {
    let (a, b) = s.split_once(',')?;
    Some((a.trim().parse().ok()?, b.trim().parse().ok()?))
}

/// Parse a `<point>`.
pub fn parse_point(s: &str) -> Option<Point> {
    if let Some(rest) = s.strip_prefix('+') {
        let (x, y) = pair(rest)?;
        return Some(Point::FromPress(x, y));
    }
    if let Some((face, rest)) = s.split_once('@') {
        let mut chars = face.chars();
        let face = chars.next().filter(|c| "URFDLB".contains(*c))?;
        if chars.next().is_some() {
            return None;
        }
        let (pos, offset) = match rest.split_once('+') {
            Some((p, o)) => (p, pair(o)?),
            None => (rest, (0.0, 0.0)),
        };
        let v: Vec<i32> = pos
            .split(',')
            .map(|x| x.trim().parse().ok())
            .collect::<Option<_>>()?;
        let pos: [i32; 3] = v.try_into().ok()?;
        return Some(Point::Sticker { face, pos, offset });
    }
    let (x, y) = pair(s)?;
    Some(Point::Pixel(x, y))
}

/// Parse a whole script.
pub fn parse(text: &str) -> Result<Vec<Command>, ParseError> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let err = |msg: &str| ParseError {
            line: i + 1,
            msg: format!("{msg}: `{line}`"),
        };
        let (word, rest) = line.split_once(char::is_whitespace).unwrap_or((line, ""));
        let rest = rest.trim();
        let args: Vec<&str> = rest.split_whitespace().collect();
        let num = |k: usize| -> Result<f32, ParseError> {
            args.get(k)
                .and_then(|a| a.parse().ok())
                .ok_or_else(|| err("expected a number"))
        };
        let point = |k: usize| -> Result<Point, ParseError> {
            args.get(k)
                .and_then(|a| parse_point(a))
                .ok_or_else(|| err("expected a point"))
        };
        let right = |k: usize| args.get(k) == Some(&"right");
        let cmd = match word {
            "cuboid" => Command::Cuboid(rest.to_string()),
            "size" => {
                let (w, h) = (num(0)?, num(1)?);
                // yuv420p video needs even dimensions.
                let even = |v: f32| v >= 2.0 && v.fract() == 0.0 && v % 2.0 == 0.0;
                if !even(w) || !even(h) {
                    return Err(err("size must be positive even integers"));
                }
                Command::Size(w as u32, h as u32)
            }
            "case" => Command::Case(rest.to_string()),
            "note" => Command::Note(rest.to_string()),
            "wait" => Command::Wait(num(0)?),
            "settle" => Command::Settle,
            "press" => Command::Press(point(0)?, right(1)),
            "move" => Command::Move(point(0)?, num(1)?),
            "release" => Command::Release,
            "drag" => {
                out.push(Command::Press(point(0)?, right(3)));
                Command::Move(point(1)?, num(2)?)
            }
            "scroll" => Command::Scroll(num(0)?),
            "moves" => Command::Moves(rest.to_string()),
            "undo" => Command::Undo,
            "reset" => Command::Reset,
            "solve" => Command::Solve,
            "scramble" => Command::Scramble(num(0)? as usize, num(1)? as u64),
            "setup" => Command::Setup(rest.to_string()),
            "home" => Command::Home,
            "solve-latency" => Command::SolveLatency(num(0)?),
            "expect" => {
                let (what, arg) = rest.split_once(char::is_whitespace).unwrap_or((rest, ""));
                let arg = arg.trim().to_string();
                Command::Expect(match what {
                    "solved" => Expect::Solved,
                    "unsolved" => Expect::Unsolved,
                    "state" => Expect::State(arg),
                    "idle" => Expect::Idle,
                    "feedback" => Expect::Feedback(arg),
                    "no-feedback" => Expect::NoFeedback(arg),
                    "camera-moved" => Expect::CameraMoved,
                    "camera-still" => Expect::CameraStill,
                    _ => return Err(err("unknown expectation")),
                })
            }
            _ => return Err(err("unknown command")),
        };
        let is_drag = word == "drag";
        out.push(cmd);
        if is_drag {
            out.push(Command::Release);
        }
    }
    Ok(out)
}

impl fmt::Display for Expect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expect::Solved => write!(f, "solved"),
            Expect::Unsolved => write!(f, "unsolved"),
            Expect::State(s) => write!(f, "state = solved + [{s}]"),
            Expect::Idle => write!(f, "idle"),
            Expect::Feedback(s) => write!(f, "feedback contains \"{s}\""),
            Expect::NoFeedback(s) => write!(f, "no feedback contains \"{s}\""),
            Expect::CameraMoved => write!(f, "camera moved"),
            Expect::CameraStill => write!(f, "camera still"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_and_drags_parse() {
        assert_eq!(parse_point("10,20"), Some(Point::Pixel(10.0, 20.0)));
        assert_eq!(parse_point("+0,-120"), Some(Point::FromPress(0.0, -120.0)));
        assert_eq!(
            parse_point("F@2,0,2+5,-5"),
            Some(Point::Sticker {
                face: 'F',
                pos: [2, 0, 2],
                offset: (5.0, -5.0)
            })
        );
        let cmds = parse("drag F@2,0,2 +0,-100 0.5 right # comment\nexpect state R").unwrap();
        assert_eq!(cmds.len(), 4);
        assert!(matches!(cmds[0], Command::Press(_, true)));
        assert_eq!(cmds[2], Command::Release);
        assert_eq!(cmds[3], Command::Expect(Expect::State("R".into())));
        assert!(parse("bogus").is_err());
        assert_eq!(parse_point("FF@2,0,2"), None);
        assert!(parse("size 640 480").is_ok());
        assert!(parse("size 641 480").is_err());
        assert!(parse("size 640 0").is_err());
    }
}
