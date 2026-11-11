//! The eframe desktop front end.
//!
//! Translates egui pointer, wheel, keyboard and button events into
//! [`Input`](gpss_viz::controller::Input)s for the shared [`Controller`], runs solves on a
//! worker thread, and shows the [`Renderer`]'s texture in the central panel.

use std::process::ExitCode;
use std::sync::mpsc;

use eframe::egui;
use eframe::wgpu;
use glam::Vec2;
use gpss_core::Puzzle;
use gpss_puzzles::cuboid::{Cuboid, Dims};
use gpss_viz::camera::OrbitCamera;
use gpss_viz::controller::{Button, Controller, Feedback, Input};
use gpss_viz::geom::half_extents;
use gpss_viz::solve::{self, SolveError, Solved};

use crate::render::Renderer;

type SolveResult = (u64, Result<Solved, SolveError>);

struct ViewerApp {
    ctl: Controller,
    renderer: Renderer,
    texture: Option<egui::TextureId>,
    solved_tx: mpsc::Sender<SolveResult>,
    solved_rx: mpsc::Receiver<SolveResult>,
    log: Vec<String>,
    dims: String,
    moves: String,
    scramble_len: usize,
    seed: u64,
    pressed: bool,
}

const FACE_KEYS: [(egui::Key, &str); 6] = [
    (egui::Key::U, "U"),
    (egui::Key::R, "R"),
    (egui::Key::F, "F"),
    (egui::Key::D, "D"),
    (egui::Key::L, "L"),
    (egui::Key::B, "B"),
];

impl ViewerApp {
    fn new(cc: &eframe::CreationContext<'_>, dims: Dims, scramble: Option<usize>) -> Self {
        let rs = cc
            .wgpu_render_state
            .as_ref()
            .expect("eframe was started with wgpu");
        let (solved_tx, solved_rx) = mpsc::channel();
        let mut app = Self {
            ctl: Controller::new(Cuboid::new(dims), Vec2::new(800.0, 600.0)),
            renderer: Renderer::new(&rs.device),
            texture: None,
            solved_tx,
            solved_rx,
            log: Vec::new(),
            dims: dims.to_string(),
            moves: String::new(),
            scramble_len: scramble.unwrap_or(25),
            seed: 1,
            pressed: false,
        };
        if let Some(length) = scramble {
            app.input(Input::Scramble { length, seed: 0 });
        }
        app
    }

    fn feedback(&mut self, fb: Vec<Feedback>) {
        for f in fb {
            self.log.push(f.to_string());
        }
        let excess = self.log.len().saturating_sub(200);
        self.log.drain(..excess);
    }

    fn input(&mut self, input: Input) {
        let fb = self.ctl.handle(input);
        self.feedback(fb);
        if let Some(job) = self.ctl.take_job() {
            let tx = self.solved_tx.clone();
            std::thread::spawn(move || {
                let result = solve::run(&job);
                let _ = tx.send((job.id, result));
            });
        }
    }

    fn controls(&mut self, ui: &mut egui::Ui) {
        ui.heading("gpss viewer");
        ui.horizontal(|ui| {
            ui.label("Cuboid");
            ui.add(egui::TextEdit::singleline(&mut self.dims).desired_width(70.0));
            if ui.button("New").clicked() {
                match self.dims.parse::<Dims>() {
                    Ok(d) => {
                        let size = self.ctl.viewport();
                        self.ctl = Controller::new(Cuboid::new(d), size);
                        self.log.push(format!("new {d}"));
                    }
                    Err(e) => self.log.push(format!("rejected: {e}")),
                }
            }
        });
        ui.horizontal(|ui| {
            ui.add(
                egui::DragValue::new(&mut self.scramble_len)
                    .range(1..=200)
                    .suffix(" moves"),
            );
            if ui.button("Scramble").clicked() {
                self.seed += 1;
                let (length, seed) = (self.scramble_len, self.seed);
                self.input(Input::Scramble { length, seed });
            }
        });
        ui.horizontal(|ui| {
            let solving = self.ctl.is_solving();
            if ui
                .button(if solving { "Solving..." } else { "Solve" })
                .clicked()
            {
                self.input(Input::Solve);
            }
            if ui.button("Undo").clicked() {
                self.input(Input::Undo);
            }
            if ui.button("Reset").clicked() {
                self.input(Input::Reset);
            }
            if ui.button("Home view").clicked() {
                self.ctl.camera = OrbitCamera::framing(half_extents(self.ctl.cuboid()));
            }
        });
        ui.horizontal(|ui| {
            let r = ui.add(
                egui::TextEdit::singleline(&mut self.moves)
                    .hint_text("R U2 2F'")
                    .desired_width(130.0),
            );
            let enter = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            if ui.button("Play").clicked() || enter {
                let m = std::mem::take(&mut self.moves);
                self.input(Input::Moves(m));
            }
        });
        ui.add(egui::Slider::new(&mut self.ctl.player.speed, 0.25..=4.0).text("speed"));
        ui.separator();
        let p = &self.ctl.player;
        let solved = self.ctl.cuboid().is_solved(p.state());
        ui.label(format!(
            "{}  |  queued {}  |  {}",
            if solved { "solved" } else { "scrambled" },
            p.pending(),
            p.animation()
                .map(|a| format!("turning {}", a.mv))
                .unwrap_or_else(|| "idle".into())
        ));
        let shown = gpss_core::puzzle::format_moves(p.shown());
        ui.label(format!("moves: {shown}"));
        ui.separator();
        ui.small("Drag a sticker: turn its layer. Drag empty space or right-drag: orbit. Wheel: zoom. Keys U R F D L B (Shift = prime), Ctrl+Z undo, S solve.");
        ui.separator();
        egui::ScrollArea::vertical()
            .stick_to_bottom(true)
            .show(ui, |ui| {
                for l in &self.log {
                    ui.monospace(l);
                }
            });
    }

    fn keyboard(&mut self, ui: &egui::Ui) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let mut inputs = Vec::new();
        ui.input(|i| {
            for (key, face) in FACE_KEYS {
                if i.key_pressed(key) && !i.modifiers.command {
                    let prime = if i.modifiers.shift { "'" } else { "" };
                    inputs.push(Input::Moves(format!("{face}{prime}")));
                }
            }
            if i.key_pressed(egui::Key::Z) && i.modifiers.command {
                inputs.push(Input::Undo);
            }
            if i.key_pressed(egui::Key::S) && !i.modifiers.command {
                inputs.push(Input::Solve);
            }
        });
        for input in inputs {
            self.input(input);
        }
    }

    fn viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let (rect, response) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        let ppp = ui.ctx().pixels_per_point();
        let size = (
            (rect.width() * ppp).round() as u32,
            (rect.height() * ppp).round() as u32,
        );
        let viewport = Vec2::new(size.0.max(1) as f32, size.1.max(1) as f32);
        if viewport != self.ctl.viewport() {
            self.input(Input::Resize { size: viewport });
        }
        let to_px = |p: egui::Pos2| Vec2::new((p.x - rect.min.x) * ppp, (p.y - rect.min.y) * ppp);

        let press_origin = ui.input(|i| i.pointer.press_origin());
        let button = if response.drag_started_by(egui::PointerButton::Secondary) {
            Some(Button::Secondary)
        } else if response.drag_started_by(egui::PointerButton::Primary) {
            Some(Button::Primary)
        } else {
            None
        };
        if let (Some(button), Some(origin)) = (button, press_origin) {
            self.pressed = true;
            self.input(Input::PointerDown {
                pos: to_px(origin),
                button,
            });
        }
        if self.pressed {
            if let Some(p) = response.interact_pointer_pos() {
                self.input(Input::PointerMove { pos: to_px(p) });
            }
            if response.drag_stopped() {
                let pos = response
                    .interact_pointer_pos()
                    .or(ui.input(|i| i.pointer.latest_pos()))
                    .map(to_px)
                    .unwrap_or_default();
                self.input(Input::PointerUp { pos });
                self.pressed = false;
            }
        } else if response.clicked() {
            if let Some(p) = response.interact_pointer_pos() {
                self.input(Input::PointerDown {
                    pos: to_px(p),
                    button: Button::Primary,
                });
                self.input(Input::PointerUp { pos: to_px(p) });
            }
        } else if let Some(p) = response.hover_pos() {
            self.input(Input::PointerMove { pos: to_px(p) });
        }
        if response.hovered() {
            let dy = ui.input(|i| i.smooth_scroll_delta.y);
            if dy != 0.0 {
                self.input(Input::Scroll { steps: dy / 50.0 });
            }
        }

        let Some(rs) = frame.wgpu_render_state() else {
            return;
        };
        let recreated = self.renderer.render(&rs.device, &rs.queue, &self.ctl, size);
        if let Some(view) = self.renderer.color_view() {
            let mut r = rs.renderer.write();
            match self.texture {
                Some(id) if recreated => r.update_egui_texture_from_wgpu_texture(
                    &rs.device,
                    view,
                    wgpu::FilterMode::Linear,
                    id,
                ),
                Some(_) => {}
                None => {
                    self.texture =
                        Some(r.register_native_texture(&rs.device, view, wgpu::FilterMode::Linear))
                }
            }
        }
        if let Some(id) = self.texture {
            let uv = egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0));
            ui.painter().image(id, rect, uv, egui::Color32::WHITE);
        }
    }
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        while let Ok((id, result)) = self.solved_rx.try_recv() {
            let fb = self.ctl.finish_solve(id, result);
            self.feedback(fb);
        }
        let dt = ui.input(|i| i.stable_dt).min(0.1);
        self.ctl.tick(dt);
        self.keyboard(ui);
        egui::Panel::left("controls")
            .resizable(false)
            .show(ui, |ui| self.controls(ui));
        egui::CentralPanel::default().show(ui, |ui| self.viewport(ui, frame));
        if !self.ctl.player.is_idle() || self.ctl.is_solving() {
            ui.ctx().request_repaint();
        }
    }
}

/// Run the desktop viewer.
pub fn run(dims: &str, scramble: Option<usize>) -> ExitCode {
    let dims: Dims = match dims.parse() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1100.0, 760.0])
            .with_title("gpss viewer"),
        ..Default::default()
    };
    let result = eframe::run_native(
        "gpss viewer",
        options,
        Box::new(move |cc| Ok(Box::new(ViewerApp::new(cc, dims, scramble)))),
    );
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
