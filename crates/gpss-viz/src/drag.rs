//! Drag-to-turn: which layer turn a drag across a sticker means.
//!
//! The drag is measured on the sticker's face plane. Its dominant in-plane axis `d` and the
//! face normal `n` fix the rotation axis `n × d`: the layer through the grabbed cubie turns
//! so that the grabbed sticker starts moving along the drag. Rules for ambiguous drags and
//! half-turn-only axes are listed in `docs/viewer.md`.

use glam::Vec3;
use gpss_puzzles::cuboid::{Axis, Cuboid, LayerTurn};

use crate::geom::axis_of;

/// How much larger the dominant in-plane component of a drag must be than the other one
/// before the drag is read as a turn. `1.5` leaves a ±11° dead zone around the diagonals.
pub const DOMINANCE: f32 = 1.5;

/// The outcome of interpreting a drag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragTurn {
    /// The drag means this layer turn.
    Turn(LayerTurn),
    /// The drag is too close to a diagonal of the face to tell which way it goes.
    Ambiguous,
}

/// Interpret dragging sticker slot `sticker` by the world-space vector `drag`.
///
/// A drag that would be a quarter turn about an axis that allows only half turns (a
/// non-square cross-section, e.g. R on a 3x3x2) becomes a half turn.
pub fn turn_for_drag(cuboid: &Cuboid, sticker: usize, drag: Vec3) -> DragTurn {
    let s = cuboid.stickers()[sticker];
    let (normal_axis, _) = axis_of(s.normal).expect("sticker normals are unit axis vectors");
    let mut comps: Vec<(Axis, f32)> = Axis::ALL
        .into_iter()
        .filter(|&a| a != normal_axis)
        .map(|a| (a, drag[a as usize]))
        .collect();
    comps.sort_by(|x, y| y.1.abs().total_cmp(&x.1.abs()));
    let (dom, along) = comps[0];
    let other = comps[1].1;
    if along == 0.0 || along.abs() < DOMINANCE * other.abs() {
        return DragTurn::Ambiguous;
    }
    let mut d = [0; 3];
    d[dom as usize] = if along > 0.0 { 1 } else { -1 };
    let n = s.normal;
    let omega = [
        n[1] * d[2] - n[2] * d[1],
        n[2] * d[0] - n[0] * d[2],
        n[0] * d[1] - n[1] * d[0],
    ];
    let (axis, sign) = axis_of(omega).expect("normal and drag axis are perpendicular");
    let layers = cuboid.dims().layers(axis) as i32;
    let layer = ((layers - 1 - s.pos[axis as usize]) / 2) as u8;
    // A positive (counter-clockwise) rotation about `sign * axis`; `turns` counts clockwise
    // quarter turns seen from the positive face.
    let turns = if !cuboid.dims().allows_quarter(axis) {
        2
    } else if sign > 0 {
        3
    } else {
        1
    };
    DragTurn::Turn(LayerTurn { axis, layer, turns })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::{in_layer, turn_rotation, vec3};
    use gpss_puzzles::cuboid::Dims;
    use proptest::prelude::*;

    fn sticker(c: &Cuboid, pos: [i32; 3], normal: [i32; 3]) -> usize {
        c.stickers()
            .iter()
            .position(|s| s.pos == pos && s.normal == normal)
            .unwrap()
    }

    fn turn(c: &Cuboid, pos: [i32; 3], normal: [i32; 3], drag: Vec3) -> String {
        match turn_for_drag(c, sticker(c, pos, normal), drag) {
            DragTurn::Turn(mv) => mv.to_string(),
            DragTurn::Ambiguous => "ambiguous".into(),
        }
    }

    /// Hand-checked against a physical cube held with F facing you and U on top.
    #[test]
    fn standard_drags_on_a_3x3x3() {
        let c = Cuboid::new("3x3x3".parse().unwrap());
        let f = [0, 0, 1];
        // Top row of F dragged left turns U clockwise; right is U'.
        assert_eq!(turn(&c, [0, 2, 2], f, Vec3::NEG_X), "U");
        assert_eq!(turn(&c, [0, 2, 2], f, Vec3::X), "U'");
        // Right column of F dragged up is R; down is R'.
        assert_eq!(turn(&c, [2, 0, 2], f, Vec3::Y), "R");
        assert_eq!(turn(&c, [2, 0, 2], f, Vec3::NEG_Y), "R'");
        // Middle column dragged up is the middle slice turned like R (`2R`).
        assert_eq!(turn(&c, [0, 0, 2], f, Vec3::Y), "2R");
        // Left column dragged down is L (= `3R'`).
        assert_eq!(turn(&c, [-2, 0, 2], f, Vec3::NEG_Y), "3R'");
        // Bottom row of F dragged right is D (= `3U'`).
        assert_eq!(turn(&c, [0, -2, 2], f, Vec3::X), "3U'");
        // Front row of U dragged right (towards R) is F.
        assert_eq!(turn(&c, [0, 2, 2], [0, 1, 0], Vec3::X), "F");
        // Front column of R dragged up is F'.
        assert_eq!(turn(&c, [2, 0, 2], [1, 0, 0], Vec3::Y), "F'");
        // Diagonal drags are ambiguous; the normal component is ignored.
        assert_eq!(
            turn(&c, [2, 2, 2], f, Vec3::new(1.0, 1.0, 0.0)),
            "ambiguous"
        );
        assert_eq!(turn(&c, [2, 2, 2], f, Vec3::new(0.0, 1.0, 50.0)), "R");
    }

    #[test]
    fn quarter_drags_on_half_turn_axes_become_half_turns() {
        let c = Cuboid::new("3x3x2".parse().unwrap());
        // 3 wide, 3 deep, 2 tall: R-L and F-B are half-turn only; U-D allows quarters.
        let f = [0, 0, 1];
        assert_eq!(turn(&c, [2, 1, 2], f, Vec3::Y), "R2");
        assert_eq!(turn(&c, [2, 1, 2], f, Vec3::NEG_Y), "R2");
        assert_eq!(turn(&c, [0, 1, 2], f, Vec3::NEG_X), "U");
        assert_eq!(turn(&c, [0, 1, 2], f, Vec3::X), "U'");
        assert_eq!(turn(&c, [2, 1, 2], [1, 0, 0], Vec3::NEG_Y), "F2");
    }

    proptest! {
        /// Whatever the face, cubie and drag, the chosen turn moves the grabbed cubie and
        /// starts moving the grabbed sticker along the drag.
        #[test]
        fn the_grabbed_sticker_moves_along_the_drag(
            a in 2u8..=5, b in 2u8..=5, c in 2u8..=5, k in 0usize..1000,
            dx in -1.0f32..1.0, dy in -1.0f32..1.0, dz in -1.0f32..1.0,
        ) {
            let cuboid = Cuboid::new(Dims::new(a, b, c));
            let i = k % cuboid.sticker_count();
            let s = cuboid.stickers()[i];
            let drag = Vec3::new(dx, dy, dz);
            if let DragTurn::Turn(mv) = turn_for_drag(&cuboid, i, drag) {
                prop_assert!(in_layer(&cuboid, mv, s.pos));
                let p = vec3(s.pos) + vec3(s.normal);
                let velocity = turn_rotation(mv, 1e-3) * p - p;
                let planar = drag - vec3(s.normal) * drag.dot(vec3(s.normal));
                if mv.turns != 2 || cuboid.dims().allows_quarter(mv.axis) {
                    prop_assert!(velocity.dot(planar) > 0.0, "{mv} {velocity} {planar}");
                }
                prop_assert!(gpss_core::Puzzle::moves(&cuboid).contains(&mv));
            }
        }
    }
}
