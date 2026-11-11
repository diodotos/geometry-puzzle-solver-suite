struct Uniforms {
    view_proj: mat4x4<f32>,
    eye: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs_main(@location(0) pos: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec4<f32>) -> VsOut {
    var o: VsOut;
    o.clip = u.view_proj * vec4<f32>(pos, 1.0);
    o.world = pos;
    o.normal = normal;
    o.color = color;
    return o;
}

@fragment
fn fs_main(i: VsOut) -> @location(0) vec4<f32> {
    // Head light slightly above the camera, plus ambient.
    let to_eye = normalize(u.eye.xyz - i.world);
    let light = normalize(to_eye + vec3<f32>(0.0, 0.35, 0.0));
    let n = normalize(i.normal);
    let diffuse = max(dot(n, light), 0.0);
    let shade = 0.38 + 0.62 * diffuse;
    return vec4<f32>(i.color.rgb * shade, i.color.a);
}
