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
@group(0) @binding(1) var chemo_in : texture_2d<f32>;

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

    // let particle_pos = agents[index].pos;

    let pos = (vertex_position) * vec2(1280.0, 800.0) * 2.0;
    // let pos = (vertex_position + particle_pos) * 2.0 + vec2(-1.0, -1.0);

    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn main_fs(
  @builtin(position) pos : vec4<f32>
) -> @location(0) vec4<f32> {
  
  let x = u32( pos.x );
  let y = u32( pos.y );

  let index = vec2<u32>(x, y);

  let texture_value = textureLoad(chemo_in, index, 0);
  return texture_value;
}
