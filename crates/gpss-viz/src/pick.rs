//! Ray picking against sticker slots.
//!
//! Picking uses the *slot* layout, which does not depend on the state or on an animation in
//! progress: a drag is always interpreted against where stickers will be once the current
//! turn has finished. See `docs/viewer.md` ("Input during an animation").

use glam::Vec3;
use gpss_puzzles::cuboid::Cuboid;

use crate::geom::vec3;

/// Where a ray met a sticker.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    /// Sticker slot index (as in [`Cuboid::stickers`]).
    pub sticker: usize,
    /// Distance along the ray.
    pub distance: f32,
    /// World point on the sticker's face plane.
    pub point: Vec3,
}

/// The nearest front-facing sticker slot hit by the ray `origin + t * dir` (`t > 0`).
///
/// A hit on the boundary between two stickers goes to the lower slot index, so the result
/// is deterministic.
pub fn pick(cuboid: &Cuboid, origin: Vec3, dir: Vec3) -> Option<Hit> {
    let mut best: Option<Hit> = None;
    for (i, s) in cuboid.stickers().iter().enumerate() {
        let n = vec3(s.normal);
        let denom = dir.dot(n);
        if denom >= -1e-6 {
            continue;
        }
        let centre = vec3(s.pos) + n;
        let t = (centre - origin).dot(n) / denom;
        if t <= 0.0 {
            continue;
        }
        let p = origin + dir * t;
        let off = (p - centre).abs();
        let inside = (0..3).all(|a| s.normal[a] != 0 || off[a] <= 1.0 + 1e-4);
        if inside && best.map_or(true, |b| t < b.distance - 1e-5) {
            best = Some(Hit {
                sticker: i,
                distance: t,
                point: p,
            });
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::OrbitCamera;
    use crate::geom::half_extents;
    use glam::Vec2;
    use gpss_puzzles::cuboid::Dims;
    use proptest::prelude::*;

    #[test]
    fn straight_on_hits_the_centre_sticker() {
        let c = Cuboid::new("3x3x3".parse().unwrap());
        let hit = pick(&c, Vec3::new(0.0, 0.0, 20.0), Vec3::NEG_Z).unwrap();
        let s = c.stickers()[hit.sticker];
        assert_eq!(s.normal, [0, 0, 1]);
        assert_eq!(s.pos, [0, 0, 2]);
        assert!((hit.distance - 17.0).abs() < 1e-4);
    }

    #[test]
    fn missing_the_puzzle_hits_nothing() {
        let c = Cuboid::new("3x3x3".parse().unwrap());
        assert!(pick(&c, Vec3::new(10.0, 0.0, 20.0), Vec3::NEG_Z).is_none());
        assert!(pick(&c, Vec3::new(0.0, 0.0, 20.0), Vec3::Z).is_none());
    }

    proptest! {
        /// Aiming the camera at any visible sticker's centre picks exactly that sticker.
        #[test]
        fn picking_a_projected_sticker_centre_returns_it(
            a in 2u8..=5, b in 2u8..=5, c in 2u8..=5,
            yaw in 0.0f32..std::f32::consts::TAU, pitch in -1.4f32..1.4, k in 0usize..1000,
        ) {
            let cuboid = Cuboid::new(Dims::new(a, b, c));
            let mut cam = OrbitCamera::framing(half_extents(&cuboid));
            cam.yaw = yaw;
            cam.pitch = pitch;
            let vp = Vec2::new(800.0, 600.0);
            let i = k % cuboid.sticker_count();
            let s = cuboid.stickers()[i];
            let centre = vec3(s.pos) + vec3(s.normal);
            // Only stickers facing the camera, and not at a grazing angle, are pickable.
            prop_assume!((cam.eye() - centre).normalize().dot(vec3(s.normal)) > 0.2);
            let px = cam.project(centre, vp).unwrap();
            let (o, d) = cam.ray(px, vp);
            let hit = pick(&cuboid, o, d).unwrap();
            prop_assert_eq!(hit.sticker, i);
        }
    }
}
