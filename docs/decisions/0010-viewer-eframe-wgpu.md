# 0010. Viewer: a renderer-independent view model, drawn with eframe + wgpu

**Status:** Accepted

## Context

The suite needs a 3D viewer for any `AxBxC` cuboid: show an arbitrary state, animate layer
turns, orbit by dragging, turn layers by dragging stickers, and play back solver
solutions. The repository owner asked for portability, ease of use, fast start-up,
consistency with the Rust workspace, and wanted to defer the desktop-vs-web choice. Every
feature must also be demonstrable headlessly: scripted inputs rendered to video on a
machine without a display.

## Decision

- **Two crates.** `gpss-viz` is a renderer-independent view model: sticker geometry,
  move-to-rotation, ray picking, drag-to-turn, the playback timeline, the input state
  machine (`Controller`) and the solver hand-off. It depends only on `glam` and the suite.
  `gpss-viewer` draws it: a desktop app on `eframe`/`egui` with a custom `wgpu` pipeline,
  and a headless recorder that drives the same `Controller` from scenario files and
  renders offscreen with the same pipeline.
- **The backend state is the only truth.** Frames are drawn from a `CuboidState`; an
  animation is a transient rotation of one layer's stickers, and when it ends the move is
  applied with `Puzzle::apply`. Tests check that the end pose of every animation is exactly
  the backend's sticker permutation, so the picture cannot drift from the model.
- **Solutions are `LayerTurn`s.** Whatever solver runs (two-phase for the 3x3x3, IDA* with
  orbit pattern databases otherwise), its output is converted to cuboid moves and queued
  through the same timeline as user moves.
- **eframe + wgpu** over Bevy (slow builds, large web bundle, frequent breaking changes),
  `three-d` (smaller maintainer base) and Three.js (a second language and toolchain). One
  codebase targets native desktop now and WebGPU/WebGL2 later; egui supplies the panels.
- **Rust version.** eframe/wgpu need a newer compiler than the workspace MSRV, so
  `gpss-viewer` declares its own `rust-version`; `gpss-viz` keeps the workspace MSRV.

## Consequences

- Interaction rules (see `docs/viewer.md`) live in `gpss-viz` and are unit-tested without
  a GPU; the drawing layer stays thin and replaceable.
- Headless recordings use a software Vulkan driver (lavapipe) on Linux: correct pictures,
  not representative frame rates.
- A web build is possible but not done: large tables (Korf) and ONNX solvers will not ship
  to the browser, and the table cache and RNG need web-specific handling.
- `gpss-puzzles` now exposes cuboid sticker geometry (`Cuboid::stickers`,
  `Cuboid::layer_level`, `Cuboid::move_permutation`).
