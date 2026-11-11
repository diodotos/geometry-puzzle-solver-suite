//! Registrations of concrete puzzles and solvers.
//!
//! This is the only part of `gpss-bench` that names concrete puzzles. To add a puzzle,
//! write a `register(&mut Registry)` function in a new module here and call it from
//! [`register_all`]; then run `gpss-bench check --update` to record its baseline. See
//! `docs/benchmarking.md` for a worked example.

pub mod cube2;
pub mod cube3;
pub mod cuboid;

use crate::target::Registry;

/// Register every puzzle known to the suite.
pub fn register_all(registry: &mut Registry) {
    cube2::register(registry);
    cube3::register(registry);
    cuboid::register(registry);
}
