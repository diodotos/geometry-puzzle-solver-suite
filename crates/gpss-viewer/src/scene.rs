//! Triangles for one frame, built from the controller's backend state.

use glam::{Quat, Vec3};
use gpss_viz::controller::Controller;
use gpss_viz::geom::{surface_cubies, vec3};

/// One vertex: world position, normal and linear RGBA colour.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// World position.
    pub pos: [f32; 3],
    /// Unit normal.
    pub normal: [f32; 3],
    /// Linear RGBA.
    pub color: [f32; 4],
}

/// sRGB face colours in the cuboid's `U R F D L B` colour order.
pub const FACE_COLOURS: [[u8; 3]; 6] = [
    [245, 245, 245], // U white
    [200, 20, 30],   // R red
    [0, 155, 72],    // F green
    [255, 213, 0],   // D yellow
    [255, 110, 0],   // L orange
    [0, 70, 173],    // B blue
];

const BODY: [u8; 3] = [18, 18, 20];
/// Half size of a sticker within its cubie face (a cubie face is 2 units wide).
const STICKER_HALF: f32 = 0.86;
/// How far stickers sit above the cubie surface, to avoid depth fighting.
const LIFT: f32 = 0.004;

fn linear(c: [u8; 3]) -> [f32; 4] {
    let f = |v: u8| (v as f32 / 255.0).powf(2.2);
    [f(c[0]), f(c[1]), f(c[2]), 1.0]
}

fn lighten(c: [u8; 3]) -> [u8; 3] {
    c.map(|v| v + ((255 - v) as f32 * 0.45) as u8)
}

fn tangents(n: Vec3) -> (Vec3, Vec3) {
    let a = if n.x.abs() > 0.5 { Vec3::Y } else { Vec3::X };
    let u = n.cross(a).normalize();
    (u, n.cross(u))
}

fn quad(out: &mut Vec<Vertex>, rot: Quat, centre: Vec3, n: Vec3, half: f32, color: [f32; 4]) {
    let (u, v) = tangents(n);
    let corner = |su: f32, sv: f32| Vertex {
        pos: (rot * (centre + (u * su + v * sv) * half)).to_array(),
        normal: (rot * n).to_array(),
        color,
    };
    let [a, b, c, d] = [
        corner(-1.0, -1.0),
        corner(1.0, -1.0),
        corner(1.0, 1.0),
        corner(-1.0, 1.0),
    ];
    out.extend([a, b, c, a, c, d]);
}

/// Every triangle of the current frame: a dark body for each visible cubie and a coloured
/// square for each sticker slot, coloured from the backend state and rotated by the
/// in-flight animation. `highlight` lightens one sticker (hover / grab feedback).
pub fn build(ctl: &Controller) -> Vec<Vertex> {
    let cuboid = ctl.cuboid();
    let player = &ctl.player;
    let state = player.state();
    let highlight = ctl.highlight();
    let mut out = Vec::new();
    let body = linear(BODY);
    for pos in surface_cubies(cuboid) {
        let rot = player.rotation_of(pos);
        let c = vec3(pos);
        for n in [
            Vec3::X,
            Vec3::NEG_X,
            Vec3::Y,
            Vec3::NEG_Y,
            Vec3::Z,
            Vec3::NEG_Z,
        ] {
            quad(&mut out, rot, c + n, n, 1.0, body);
        }
    }
    for (i, s) in cuboid.stickers().iter().enumerate() {
        let rot = player.rotation_of(s.pos);
        let n = vec3(s.normal);
        let mut rgb = FACE_COLOURS[state[i] as usize];
        if highlight == Some(i) {
            rgb = lighten(rgb);
        }
        let centre = vec3(s.pos) + n * (1.0 + LIFT);
        quad(&mut out, rot, centre, n, STICKER_HALF, linear(rgb));
    }
    out
}
