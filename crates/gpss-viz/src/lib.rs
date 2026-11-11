//! Renderer-independent view model for the puzzle viewer (`gpss-viewer`).
//!
//! Everything a front end needs that is not drawing pixels lives here, so it can be tested
//! without a GPU and shared by the desktop app and the headless recorder:
//!
//! * [`geom`]: sticker geometry of a [`Cuboid`](gpss_puzzles::cuboid::Cuboid) in world
//!   space and the rigid motion that animates a [`LayerTurn`](gpss_puzzles::cuboid::LayerTurn).
//! * [`camera`]: an orbit camera with projection and picking rays.
//! * [`pick`]: which sticker slot a ray hits.
//! * [`drag`]: which layer turn a drag across a sticker means.
//! * [`player`]: the logical move timeline and the animation that catches the picture up
//!   with it.
//! * [`controller`]: the input state machine (pointer, keys, commands) on top of all of
//!   the above.
//! * [`solve`]: turning a viewer state into a solver call and the solution back into layer
//!   turns.
//!
//! The rule that keeps the picture honest: **the screen is always drawn from a backend
//! [`CuboidState`](gpss_puzzles::cuboid::CuboidState)**. An animation is only a transient
//! rotation of one layer's stickers; when it ends the move is applied to the state with
//! [`Puzzle::apply`](gpss_core::Puzzle::apply) and the rotation disappears. A test checks
//! that the end pose of every animation is exactly the backend's sticker permutation.
//! Design notes: `docs/decisions/0010-viewer-eframe-wgpu.md` and `docs/viewer.md`.

pub mod camera;
pub mod controller;
pub mod drag;
pub mod geom;
pub mod pick;
pub mod player;
pub mod solve;
