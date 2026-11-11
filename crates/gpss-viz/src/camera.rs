//! Orbit camera: the viewer looks at the puzzle's centre from a point on a sphere.
//!
//! Orbiting is how the user chooses which faces are visible; the puzzle itself never
//! rotates, so sticker slots keep fixed world positions and picking stays simple.

use glam::{Mat4, Vec2, Vec3, Vec4Swizzles};

/// Highest and lowest allowed pitch (radians). Stopping short of the poles keeps the up
/// vector well defined, so the view never flips.
pub const MAX_PITCH: f32 = 85.0 * std::f32::consts::PI / 180.0;
/// Radians of orbit per pixel of drag.
pub const ORBIT_SPEED: f32 = 0.01;

/// A perspective camera on a sphere around the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    /// Rotation about world `+Y` (radians); 0 looks at the F face, positive moves the
    /// eye towards R.
    pub yaw: f32,
    /// Elevation (radians); positive looks down on U. Clamped to `±MAX_PITCH`.
    pub pitch: f32,
    /// Distance from the origin.
    pub distance: f32,
    /// Vertical field of view (radians).
    pub fov_y: f32,
}

impl OrbitCamera {
    /// The default view of a puzzle with the given half extents: U, F and R visible, the
    /// whole puzzle in frame.
    pub fn framing(half_extents: Vec3) -> Self {
        let fov_y = 35f32.to_radians();
        let radius = half_extents.length();
        Self {
            yaw: 35f32.to_radians(),
            pitch: 28f32.to_radians(),
            distance: radius / (fov_y / 2.0).sin() * 1.05,
            fov_y,
        }
    }

    /// Eye position in world space.
    pub fn eye(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cp * sy, sp, cp * cy) * self.distance
    }

    /// World-to-clip transform (depth in `0..1`, as wgpu expects).
    pub fn view_proj(&self, viewport: Vec2) -> Mat4 {
        let aspect = (viewport.x / viewport.y.max(1.0)).max(1e-3);
        let near = (self.distance * 0.05).max(0.01);
        let far = self.distance * 4.0;
        Mat4::perspective_rh(self.fov_y, aspect, near, far)
            * Mat4::look_at_rh(self.eye(), Vec3::ZERO, Vec3::Y)
    }

    /// The picking ray through a pixel (`(0, 0)` is the top-left corner): origin and unit
    /// direction in world space.
    pub fn ray(&self, pixel: Vec2, viewport: Vec2) -> (Vec3, Vec3) {
        let inv = self.view_proj(viewport).inverse();
        let ndc = Vec2::new(
            2.0 * pixel.x / viewport.x - 1.0,
            1.0 - 2.0 * pixel.y / viewport.y,
        );
        let near = inv * ndc.extend(0.0).extend(1.0);
        let far = inv * ndc.extend(1.0).extend(1.0);
        let near = near.xyz() / near.w;
        let far = far.xyz() / far.w;
        (near, (far - near).normalize())
    }

    /// The pixel a world point projects to, or `None` behind the camera.
    pub fn project(&self, world: Vec3, viewport: Vec2) -> Option<Vec2> {
        let clip = self.view_proj(viewport) * world.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = clip.xyz() / clip.w;
        Some(Vec2::new(
            (ndc.x + 1.0) / 2.0 * viewport.x,
            (1.0 - ndc.y) / 2.0 * viewport.y,
        ))
    }

    /// Orbit by a pointer drag of `delta` pixels, as if grabbing the puzzle: dragging right
    /// turns the puzzle to the right, dragging down tips its top towards the viewer.
    pub fn orbit(&mut self, delta: Vec2) {
        self.yaw -= delta.x * ORBIT_SPEED;
        self.pitch = (self.pitch + delta.y * ORBIT_SPEED).clamp(-MAX_PITCH, MAX_PITCH);
        self.yaw = self.yaw.rem_euclid(std::f32::consts::TAU);
    }

    /// Zoom by `steps` (positive = closer), within `[min, max]` distance.
    pub fn zoom(&mut self, steps: f32, min: f32, max: f32) {
        self.distance = (self.distance * 0.9f32.powf(steps)).clamp(min, max);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_and_ray_round_trip() {
        let cam = OrbitCamera::framing(Vec3::splat(3.0));
        let vp = Vec2::new(800.0, 600.0);
        for p in [
            Vec3::ZERO,
            Vec3::new(3.0, 3.0, 3.0),
            Vec3::new(-2.0, 1.0, 3.0),
        ] {
            let px = cam.project(p, vp).unwrap();
            let (o, d) = cam.ray(px, vp);
            let closest = o + d * (p - o).dot(d);
            assert!((closest - p).length() < 1e-3, "{p} vs {closest}");
        }
    }

    #[test]
    fn origin_is_at_the_centre_of_the_viewport() {
        let cam = OrbitCamera::framing(Vec3::splat(3.0));
        let px = cam.project(Vec3::ZERO, Vec2::new(640.0, 480.0)).unwrap();
        assert!((px - Vec2::new(320.0, 240.0)).length() < 1e-3);
    }

    #[test]
    fn default_view_shows_u_f_r() {
        let eye = OrbitCamera::framing(Vec3::splat(3.0)).eye();
        assert!(eye.x > 0.0 && eye.y > 0.0 && eye.z > 0.0);
    }

    #[test]
    fn pitch_is_clamped() {
        let mut cam = OrbitCamera::framing(Vec3::splat(3.0));
        cam.orbit(Vec2::new(0.0, 1e6));
        assert_eq!(cam.pitch, MAX_PITCH);
        cam.orbit(Vec2::new(0.0, -1e6));
        assert_eq!(cam.pitch, -MAX_PITCH);
    }

    #[test]
    fn dragging_right_brings_the_left_side_into_view() {
        let mut cam = OrbitCamera::framing(Vec3::splat(3.0));
        let before = cam.eye().x;
        cam.orbit(Vec2::new(100.0, 0.0));
        assert!(cam.eye().x < before);
    }
}
