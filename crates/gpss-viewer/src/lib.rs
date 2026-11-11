//! 3D viewer for gpss cuboids (ADR 0010).
//!
//! [`render`] draws a [`gpss_viz::controller::Controller`] with wgpu; [`app`] is the eframe
//! desktop front end and [`record`] the headless one that plays [`scenario`] scripts to
//! video. Both front ends feed the same controller, so behaviour is identical.

pub mod app;
pub mod record;
pub mod render;
pub mod scenario;
pub mod scene;
