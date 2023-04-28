struct Particle {
  pos : vec2<f32>,
  heading : f32,
  padding : f32,
};

struct SimParams {
  deltaT : f32,
  rule1Distance : f32,
  rule2Distance : f32,
  rule3Distance : f32,
  rule1Scale : f32,
  rule2Scale : f32,
  rule3Scale : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var<storage, read> agents : array<Particle>;

@vertex
fn main_vs(
    @location(0) _particle_pos: vec2<f32>,
    @location(1) particle_heading: f32,
    @location(2) vertex_position: vec2<f32>,
    @builtin(instance_index) index: u32

) -> @builtin(position) vec4<f32> {
    // let angle = -atan2(particle_vel.x, particle_vel.y);
    // let pos = vec2<f32>(
    //     position.x * cos(angle) - position.y * sin(angle),
    //     position.x * sin(angle) + position.y * cos(angle)
    // );

    let particle_pos = agents[index].pos;

    let pos = (vertex_position + particle_pos) * 2.0 + vec2(-1.0, -1.0);
    // let pos = (vertex_position + particle_pos) + vec2(-0.5, -0.5);

    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn main_fs() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}
