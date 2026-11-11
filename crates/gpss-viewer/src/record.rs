//! Headless front end: plays a [`scenario`](crate::scenario) script through the shared
//! [`Controller`], renders every frame offscreen and encodes a captioned MP4 with ffmpeg.
//!
//! Each run also writes `<name>.log`: every input, every [`Feedback`] with its time, and
//! the verdict of every `expect`. The cursor is drawn into the frames (orange while the
//! primary button is down, magenta for the secondary button, with a line back to where
//! the press started); captions, feedback and status are burnt in as ASS subtitles.

use std::fmt::Write as _;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Child, Command as Process, Stdio};

use eframe::wgpu;
use glam::Vec2;
use gpss_core::Puzzle;
use gpss_puzzles::cuboid::{Cuboid, Dims};
use gpss_viz::camera::OrbitCamera;
use gpss_viz::controller::{Button, Controller, Feedback, Input};
use gpss_viz::geom::vec3;
use gpss_viz::solve::{self, SolveError, SolveJob, Solved};

use crate::render::Renderer;
use crate::scenario::{self, Command, Expect, Point};

/// Recorder settings.
#[derive(Clone, Debug)]
pub struct Options {
    /// Directory for `<name>.mp4` and `<name>.log`.
    pub out_dir: PathBuf,
    /// Frames per second.
    pub fps: u32,
}

/// Outcome of one script.
#[derive(Clone, Debug)]
pub struct Report {
    /// Script name (file stem).
    pub name: String,
    /// Expectations checked.
    pub checked: usize,
    /// Expectations that failed, with their case.
    pub failures: Vec<String>,
    /// The video.
    pub video: PathBuf,
    /// The log.
    pub log: PathBuf,
}

const FACES: [(char, [i32; 3]); 6] = [
    ('U', [0, 1, 0]),
    ('R', [1, 0, 0]),
    ('F', [0, 0, 1]),
    ('D', [0, -1, 0]),
    ('L', [-1, 0, 0]),
    ('B', [0, 0, -1]),
];

struct Rec<'a> {
    device: &'a wgpu::Device,
    queue: &'a wgpu::Queue,
    renderer: &'a mut Renderer,
    ctl: Controller,
    size: (u32, u32),
    fps: u32,
    frame: u64,
    pointer: Vec2,
    press: Option<(Vec2, Button)>,
    latency: f32,
    solves: Vec<(f32, u64, Result<Solved, SolveError>)>,
    ffmpeg: Option<Child>,
    subs: Vec<(f32, f32, &'static str, String)>,
    status: Option<(f32, String)>,
    log: String,
    case: String,
    note: String,
    case_start: f32,
    case_feedback: Vec<String>,
    case_camera: OrbitCamera,
    checked: usize,
    failures: Vec<String>,
}

fn ass_time(t: f32) -> String {
    let cs = (t.max(0.0) * 100.0).round() as u64;
    format!(
        "{}:{:02}:{:02}.{:02}",
        cs / 360_000,
        cs / 6000 % 60,
        cs / 100 % 60,
        cs % 100
    )
}

fn ass_escape(s: &str) -> String {
    s.replace('\\', "/").replace('{', "(").replace('}', ")")
}

impl Rec<'_> {
    fn t(&self) -> f32 {
        self.frame as f32 / self.fps as f32
    }

    fn viewport(&self) -> Vec2 {
        Vec2::new(self.size.0 as f32, self.size.1 as f32)
    }

    fn say(&mut self, style: &'static str, text: String, secs: f32) {
        let t = self.t();
        self.subs.push((t, t + secs, style, text));
    }

    fn input(&mut self, input: Input) {
        let label = match &input {
            Input::PointerMove { .. } => None,
            other => Some(format!("{other:?}")),
        };
        if let Some(l) = &label {
            let _ = writeln!(self.log, "{:7.3}s  input    {l}", self.t());
        }
        let solve = matches!(input, Input::Solve);
        let fb = self.ctl.handle(input);
        self.feedback(fb);
        if solve {
            if let Some(job) = self.ctl.take_job() {
                self.start_solve(job);
            }
        }
    }

    fn start_solve(&mut self, job: SolveJob) {
        let result = solve::run(&job);
        self.solves.push((self.t() + self.latency, job.id, result));
    }

    fn feedback(&mut self, fb: Vec<Feedback>) {
        for f in fb {
            let text = f.to_string();
            let _ = writeln!(self.log, "{:7.3}s  feedback {text}", self.t());
            self.case_feedback.push(text.clone());
            self.say("Feedback", text, 2.5);
        }
    }

    fn status_text(&self) -> String {
        let p = &self.ctl.player;
        let anim = p
            .animation()
            .map(|a| {
                format!(
                    "animating {}{}",
                    a.mv,
                    if a.reverting { " (undo)" } else { "" }
                )
            })
            .unwrap_or_else(|| "idle".into());
        let solved = if self.ctl.cuboid().is_solved(p.state()) {
            "solved"
        } else {
            "scrambled"
        };
        let solving = if self.ctl.is_solving() {
            "  solving"
        } else {
            ""
        };
        format!(
            "t={:.1}s  {}  {anim}  queued {}  shown {}{solving}",
            self.t(),
            solved,
            p.pending(),
            p.shown().len()
        )
    }

    fn flush_status(&mut self, force: bool) {
        let text = self.status_text();
        match &self.status {
            Some((_, s)) if *s == text && !force => {}
            _ => {
                let t = self.t();
                if let Some((t0, s)) = self.status.take() {
                    self.subs.push((t0, t, "Status", s));
                }
                self.status = Some((t, text));
            }
        }
    }

    fn step(&mut self) -> Result<(), String> {
        let now = self.t();
        let due: Vec<_> = {
            let (due, rest) = std::mem::take(&mut self.solves)
                .into_iter()
                .partition(|(t, _, _)| *t <= now + 1e-6);
            self.solves = rest;
            due
        };
        for (_, id, result) in due {
            let fb = self.ctl.finish_solve(id, result);
            self.feedback(fb);
        }
        self.flush_status(false);
        self.renderer
            .render(self.device, self.queue, &self.ctl, self.size);
        let mut rgba = self.renderer.read_back(self.device, self.queue);
        self.draw_cursor(&mut rgba);
        let stdin = self
            .ffmpeg
            .as_mut()
            .and_then(|c| c.stdin.as_mut())
            .ok_or("ffmpeg closed")?;
        stdin.write_all(&rgba).map_err(|e| format!("ffmpeg: {e}"))?;
        self.frame += 1;
        self.ctl.tick(1.0 / self.fps as f32);
        Ok(())
    }

    fn frames(&mut self, secs: f32) -> Result<(), String> {
        for _ in 0..(secs * self.fps as f32).round() as u32 {
            self.step()?;
        }
        Ok(())
    }

    fn resolve(&self, p: &Point) -> Result<Vec2, String> {
        match p {
            Point::Pixel(x, y) => Ok(Vec2::new(*x, *y)),
            Point::FromPress(dx, dy) => {
                let origin = self.press.map(|(s, _)| s).unwrap_or(self.pointer);
                Ok(origin + Vec2::new(*dx, *dy))
            }
            Point::Sticker { face, pos, offset } => {
                let normal = FACES
                    .iter()
                    .find(|(c, _)| c == face)
                    .map(|(_, n)| *n)
                    .unwrap();
                if !self
                    .ctl
                    .cuboid()
                    .stickers()
                    .iter()
                    .any(|s| s.pos == *pos && s.normal == normal)
                {
                    return Err(format!("no sticker {face}@{pos:?}"));
                }
                let world = vec3(*pos) + vec3(normal);
                let px = self
                    .ctl
                    .camera
                    .project(world, self.viewport())
                    .ok_or_else(|| format!("sticker {face}@{pos:?} is behind the camera"))?;
                Ok(px + Vec2::new(offset.0, offset.1))
            }
        }
    }

    fn check(&mut self, e: &Expect) {
        let c = self.ctl.cuboid();
        let state = self.ctl.player.state();
        let ok = match e {
            Expect::Solved => c.is_solved(state),
            Expect::Unsolved => !c.is_solved(state),
            Expect::State(moves) => match c.parse_sequence(moves) {
                Ok(m) => c.scrambled(&m) == *state,
                Err(_) => false,
            },
            Expect::Idle => self.ctl.player.is_idle() && !self.ctl.is_solving(),
            Expect::Feedback(s) => self.case_feedback.iter().any(|f| f.contains(s.as_str())),
            Expect::NoFeedback(s) => !self.case_feedback.iter().any(|f| f.contains(s.as_str())),
            Expect::CameraMoved => {
                (
                    self.ctl.camera.yaw,
                    self.ctl.camera.pitch,
                    self.ctl.camera.distance,
                ) != (
                    self.case_camera.yaw,
                    self.case_camera.pitch,
                    self.case_camera.distance,
                )
            }
            Expect::CameraStill => {
                (
                    self.ctl.camera.yaw,
                    self.ctl.camera.pitch,
                    self.ctl.camera.distance,
                ) == (
                    self.case_camera.yaw,
                    self.case_camera.pitch,
                    self.case_camera.distance,
                )
            }
        };
        self.checked += 1;
        let verdict = if ok { "PASS" } else { "FAIL" };
        let _ = writeln!(self.log, "{:7.3}s  {verdict}     expect {e}", self.t());
        if !ok {
            self.failures.push(format!("{}: expect {e}", self.case));
        }
        self.say(
            if ok { "Pass" } else { "Fail" },
            format!("{verdict}: expect {e}"),
            2.0,
        );
    }

    fn caption(&mut self) {
        let t = self.t();
        if let Some(last) = self
            .subs
            .iter_mut()
            .rev()
            .find(|s| s.2 == "Caption" && s.1 > t)
        {
            last.1 = t;
        }
        let text = if self.note.is_empty() {
            self.case.clone()
        } else {
            format!(
                "{}\\N{{\\fs15\\c&HC8C8C8&}}{}",
                ass_escape(&self.case),
                ass_escape(&self.note)
            )
        };
        self.subs.push((t, f32::INFINITY, "Caption", text));
    }

    fn run(&mut self, cmds: &[Command]) -> Result<(), String> {
        for cmd in cmds {
            match cmd {
                Command::Cuboid(_) | Command::Size(..) => {}
                Command::Case(title) => {
                    let _ = writeln!(self.log, "\n{:7.3}s  CASE     {title}", self.t());
                    self.case = title.clone();
                    self.note.clear();
                    self.case_start = self.t();
                    self.case_feedback.clear();
                    self.case_camera = self.ctl.camera;
                    self.caption();
                }
                Command::Note(text) => {
                    let _ = writeln!(self.log, "{:7.3}s  note     {text}", self.t());
                    self.note = text.clone();
                    self.caption();
                }
                Command::Wait(s) => self.frames(*s)?,
                Command::Settle => {
                    let limit = self.frame + 120 * self.fps as u64;
                    while !(self.ctl.player.is_idle() && self.solves.is_empty()) {
                        if self.frame > limit {
                            return Err("settle: still busy after 120 s".into());
                        }
                        self.step()?;
                    }
                }
                Command::Press(p, right) => {
                    let pos = self.resolve(p)?;
                    let button = if *right {
                        Button::Secondary
                    } else {
                        Button::Primary
                    };
                    self.pointer = pos;
                    self.input(Input::PointerMove { pos });
                    self.input(Input::PointerDown { pos, button });
                    if self.press.is_none() {
                        self.press = Some((pos, button));
                    }
                }
                Command::Move(p, secs) => {
                    let to = self.resolve(p)?;
                    let from = self.pointer;
                    let n = ((secs * self.fps as f32).round() as u32).max(1);
                    for k in 1..=n {
                        let pos = from.lerp(to, k as f32 / n as f32);
                        self.pointer = pos;
                        self.input(Input::PointerMove { pos });
                        self.step()?;
                    }
                }
                Command::Release => {
                    let pos = self.pointer;
                    self.input(Input::PointerUp { pos });
                    self.press = None;
                }
                Command::Scroll(s) => self.input(Input::Scroll { steps: *s }),
                Command::Moves(m) => self.input(Input::Moves(m.clone())),
                Command::Undo => self.input(Input::Undo),
                Command::Reset => self.input(Input::Reset),
                Command::Solve => self.input(Input::Solve),
                Command::Scramble(len, seed) => self.input(Input::Scramble {
                    length: *len,
                    seed: *seed,
                }),
                Command::Setup(m) => self.input(Input::Setup(m.clone())),
                Command::SolveLatency(s) => self.latency = *s,
                Command::Expect(e) => self.check(e),
            }
        }
        self.frames(0.5)?;
        self.flush_status(true);
        Ok(())
    }

    fn draw_cursor(&self, rgba: &mut [u8]) {
        let (w, h) = (self.size.0 as i32, self.size.1 as i32);
        let mut put = |x: i32, y: i32, c: [u8; 3]| {
            if (0..w).contains(&x) && (0..h).contains(&y) {
                let i = ((y * w + x) * 4) as usize;
                rgba[i..i + 3].copy_from_slice(&c);
            }
        };
        let p = self.pointer;
        if let Some((start, _)) = self.press {
            let n = (p.distance(start) as i32).max(1);
            for k in 0..=n {
                let q = start.lerp(p, k as f32 / n as f32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    put(q.x as i32 + dx, q.y as i32 + dy, [255, 230, 60]);
                }
            }
        }
        let fill = match self.press {
            Some((_, Button::Primary)) => [255, 140, 0],
            Some((_, Button::Secondary)) => [230, 40, 230],
            None => [255, 255, 255],
        };
        let r = 9.0;
        for dy in -11..=11 {
            for dx in -11..=11 {
                let d = ((dx * dx + dy * dy) as f32).sqrt();
                let c = if d <= r - 2.5 {
                    fill
                } else if d <= r {
                    [0, 0, 0]
                } else {
                    continue;
                };
                put(p.x as i32 + dx, p.y as i32 + dy, c);
            }
        }
    }

    fn ass(&self) -> String {
        let (w, h) = self.size;
        let mut s = format!(
            "[Script Info]\nScriptType: v4.00+\nPlayResX: {w}\nPlayResY: {h}\nWrapStyle: 0\n\n\
             [V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n\
             Style: Caption,DejaVu Sans,20,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,4,0,7,12,12,10,1\n\
             Style: Feedback,DejaVu Sans Mono,14,&H0080FFFF,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,3,3,0,1,12,12,34,1\n\
             Style: Pass,DejaVu Sans Mono,14,&H0060FF60,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,3,0,9,12,12,80,1\n\
             Style: Fail,DejaVu Sans Mono,16,&H004040FF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,3,0,9,12,12,80,1\n\
             Style: Status,DejaVu Sans Mono,12,&H00D0D0D0,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,3,3,0,1,12,12,8,1\n\n\
             [Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        );
        let end = self.t();
        for (t0, t1, style, text) in &self.subs {
            let text = if *style == "Caption" {
                text.clone()
            } else {
                ass_escape(text)
            };
            let _ = writeln!(
                s,
                "Dialogue: 0,{},{},{style},,0,0,0,,{text}",
                ass_time(*t0),
                ass_time(t1.min(end))
            );
        }
        s
    }
}

/// Record one script.
pub fn record(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    renderer: &mut Renderer,
    script: &Path,
    opts: &Options,
) -> Result<Report, String> {
    let text = std::fs::read_to_string(script).map_err(|e| format!("{}: {e}", script.display()))?;
    let cmds = scenario::parse(&text).map_err(|e| format!("{}: {e}", script.display()))?;
    let mut dims = Dims::new(3, 3, 3);
    let mut size = (640, 480);
    for c in &cmds {
        match c {
            Command::Cuboid(d) => dims = d.parse().map_err(|e| format!("{e}"))?,
            Command::Size(w, h) => size = (*w, *h),
            _ => {}
        }
    }
    let name = script
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    std::fs::create_dir_all(&opts.out_dir).map_err(|e| e.to_string())?;
    let video = opts.out_dir.join(format!("{name}.mp4"));
    let raw = opts.out_dir.join(format!("{name}.raw.mp4"));
    let ass_path = opts.out_dir.join(format!("{name}.ass"));
    let log_path = opts.out_dir.join(format!("{name}.log"));
    let ffmpeg = Process::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
        ])
        .arg(format!("{}x{}", size.0, size.1))
        .args([
            "-r",
            &opts.fps.to_string(),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
        ])
        .args(["-crf", "18", "-pix_fmt", "yuv420p"])
        .arg(&raw)
        .stdin(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot start ffmpeg: {e}"))?;
    let ctl = Controller::new(Cuboid::new(dims), Vec2::new(size.0 as f32, size.1 as f32));
    let camera = ctl.camera;
    let mut rec = Rec {
        device,
        queue,
        renderer,
        ctl,
        size,
        fps: opts.fps,
        frame: 0,
        pointer: Vec2::new(size.0 as f32 * 0.9, size.1 as f32 * 0.9),
        press: None,
        latency: 0.0,
        solves: Vec::new(),
        ffmpeg: Some(ffmpeg),
        subs: Vec::new(),
        status: None,
        log: format!(
            "scenario {name} ({dims}, {}x{} @ {} fps)\n",
            size.0, size.1, opts.fps
        ),
        case: String::new(),
        note: String::new(),
        case_start: 0.0,
        case_feedback: Vec::new(),
        case_camera: camera,
        checked: 0,
        failures: Vec::new(),
    };
    let result = rec.run(&cmds);
    let mut child = rec.ffmpeg.take().expect("ffmpeg handle");
    drop(child.stdin.take());
    let status = child.wait().map_err(|e| e.to_string())?;
    if let Err(e) = result {
        let _ = writeln!(rec.log, "ERROR: {e}");
        std::fs::write(&log_path, &rec.log).map_err(|e| e.to_string())?;
        return Err(format!("{name}: {e}"));
    }
    if !status.success() {
        return Err(format!("{name}: ffmpeg failed"));
    }
    std::fs::write(&ass_path, rec.ass()).map_err(|e| e.to_string())?;
    let burn = Process::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&raw)
        .arg("-vf")
        .arg(format!("ass={}", ass_path.display()))
        .args([
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        ])
        .arg(&video)
        .status()
        .map_err(|e| e.to_string())?;
    if !burn.success() {
        return Err(format!("{name}: ffmpeg subtitle pass failed"));
    }
    let _ = std::fs::remove_file(&raw);
    let _ = writeln!(
        rec.log,
        "\n{} expectations, {} failed",
        rec.checked,
        rec.failures.len()
    );
    std::fs::write(&log_path, &rec.log).map_err(|e| e.to_string())?;
    Ok(Report {
        name,
        checked: rec.checked,
        failures: rec.failures,
        video,
        log: log_path,
    })
}
