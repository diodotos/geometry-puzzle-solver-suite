//! World-space geometry of a cuboid and of a layer turn in progress.
//!
//! World coordinates are the cuboid's doubled integer coordinates as `f32`: a cubie centred
//! at `pos` spans `pos ± 1`, so a sticker's face plane sits at `pos + normal`. `+X` points
//! to R, `+Y` to U and `+Z` to F (right-handed, `Y` up).

use std::f32::consts::FRAC_PI_2;

use glam::{Quat, Vec3};
use gpss_puzzles::cuboid::{Axis, Cuboid, LayerTurn};

/// `v` as a float vector.
pub fn vec3(v: [i32; 3]) -> Vec3 {
    Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32)
}

/// Unit vector of the positive direction of `axis`.
pub fn axis_vec(axis: Axis) -> Vec3 {
    vec3(axis.unit())
}

/// The axis a unit vector points along, and the sign of the direction.
pub fn axis_of(v: [i32; 3]) -> Option<(Axis, i32)> {
    let mut found = None;
    for a in Axis::ALL {
        match v[a as usize] {
            0 => {}
            s if found.is_none() && s.abs() == 1 => found = Some((a, s)),
            _ => return None,
        }
    }
    found
}

/// A layer turn as signed quarter turns of a right-handed rotation about the positive axis.
///
/// [`LayerTurn::turns`] counts clockwise quarter turns seen from the positive face, which
/// is a *negative* angle about the axis. A three-quarter turn is animated the short way, as
/// one counter-clockwise quarter.
pub fn signed_quarters(mv: LayerTurn) -> i32 {
    match mv.turns {
        1 => -1,
        2 => -2,
        3 => 1,
        t => panic!("LayerTurn::turns must be 1, 2 or 3, got {t}"),
    }
}

/// Rotation of the turning layer after `fraction` (0 to 1) of the animation of `mv`.
pub fn turn_rotation(mv: LayerTurn, fraction: f32) -> Quat {
    Quat::from_axis_angle(
        axis_vec(mv.axis),
        signed_quarters(mv) as f32 * FRAC_PI_2 * fraction,
    )
}

/// Whether the cubie centred at `pos` moves with `mv`.
pub fn in_layer(cuboid: &Cuboid, mv: LayerTurn, pos: [i32; 3]) -> bool {
    pos[mv.axis as usize] == cuboid.layer_level(mv.axis, mv.layer)
}

/// Half extents of the whole puzzle in world units (`layers` along each axis).
pub fn half_extents(cuboid: &Cuboid) -> Vec3 {
    let d = cuboid.dims().0;
    Vec3::new(d[0] as f32, d[1] as f32, d[2] as f32)
}

/// Smoothstep easing used by the animation: slow start and stop, symmetric.
pub fn ease(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Every distinct cubie centre that carries at least one sticker, in first-seen sticker
/// order. Interior cubies are never visible, so a renderer only needs these.
pub fn surface_cubies(cuboid: &Cuboid) -> Vec<[i32; 3]> {
    let mut seen = std::collections::HashSet::new();
    cuboid
        .stickers()
        .iter()
        .filter(|s| seen.insert(s.pos))
        .map(|s| s.pos)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpss_puzzles::cuboid::Dims;
    use proptest::prelude::*;
    use std::collections::HashMap;

    fn round(v: Vec3) -> [i32; 3] {
        [v.x.round() as i32, v.y.round() as i32, v.z.round() as i32]
    }

    /// The end pose of the animation of every legal move is exactly the backend's sticker
    /// permutation: the sticker in slot `i` lands in slot `j` with `perm[j] == i`, and
    /// stickers outside the layer stay put. This is what makes "draw the state, animate on
    /// top" consistent.
    fn check_animation_matches_permutation(dims: Dims) {
        let cuboid = Cuboid::new(dims);
        let index: HashMap<([i32; 3], [i32; 3]), usize> = cuboid
            .stickers()
            .iter()
            .enumerate()
            .map(|(i, s)| ((s.pos, s.normal), i))
            .collect();
        for &mv in gpss_core::Puzzle::moves(&cuboid) {
            let perm = cuboid.move_permutation(mv);
            let q = turn_rotation(mv, 1.0);
            for (i, s) in cuboid.stickers().iter().enumerate() {
                if in_layer(&cuboid, mv, s.pos) {
                    let j = index[&(round(q * vec3(s.pos)), round(q * vec3(s.normal)))];
                    assert_eq!(perm[j] as usize, i, "{dims} {mv}: sticker {i}");
                } else {
                    assert_eq!(perm[i] as usize, i, "{dims} {mv}: sticker {i} moved");
                }
            }
        }
    }

    #[test]
    fn animation_end_pose_is_the_backend_permutation_fixed() {
        for d in ["2x2x2", "3x3x3", "4x4x4", "3x3x2", "2x3x4", "5x5x5"] {
            check_animation_matches_permutation(d.parse().unwrap());
        }
    }

    proptest! {
        #[test]
        fn animation_end_pose_is_the_backend_permutation(a in 2u8..=6, b in 2u8..=6, c in 2u8..=6) {
            check_animation_matches_permutation(Dims::new(a, b, c));
        }

        #[test]
        fn animation_starts_at_identity(a in 2u8..=5, b in 2u8..=5, c in 2u8..=5, k in 0usize..64) {
            let cuboid = Cuboid::new(Dims::new(a, b, c));
            let moves = gpss_core::Puzzle::moves(&cuboid);
            let mv = moves[k % moves.len()];
            prop_assert!(turn_rotation(mv, 0.0).abs_diff_eq(Quat::IDENTITY, 1e-6));
        }
    }

    #[test]
    fn surface_cubie_counts() {
        let count = |d: &str| surface_cubies(&Cuboid::new(d.parse().unwrap())).len();
        assert_eq!(count("2x2x2"), 8);
        assert_eq!(count("3x3x3"), 26);
        assert_eq!(count("4x4x4"), 56);
        assert_eq!(count("3x3x2"), 18);
    }

    #[test]
    fn easing_is_monotone_and_pinned() {
        assert_eq!(ease(0.0), 0.0);
        assert_eq!(ease(1.0), 1.0);
        let mut prev = 0.0;
        for i in 1..=100 {
            let e = ease(i as f32 / 100.0);
            assert!(e >= prev);
            prev = e;
        }
    }
}
