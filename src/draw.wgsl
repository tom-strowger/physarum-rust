struct Particle {
  pos : vec2<f32>,
  heading : f32,
  padding : f32,
};

struct SimParams {
  background_colour: vec4<f32>,  // RGBA
  foreground_colour: vec4<f32>,  // RGBA
  sense_angle : f32,
  sense_offset : f32,
  step : f32,
  rotate_angle : f32,
  max_chemo : f32,
  deposit_chemo : f32,
  decay_chemo : f32,
  sim_width : u32,
  sim_height : u32,
  control_alpha : f32,
  num_agents: u32,
  chemo_squared_detractor: f32
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var chemo_in : texture_2d<f32>;
@group(0) @binding(2) var control_texture : texture_2d<f32>;

@vertex
fn main_vs(
    @location(0) _particle_pos: vec2<f32>,
    @location(1) particle_heading: f32,
    @location(2) vertex_position: vec2<f32>,
    @builtin(instance_index) index: u32

) -> @builtin(position) vec4<f32> {

    let pos = (vertex_position) * vec2<f32>(vec2(params.sim_width, params.sim_height));
    return vec4<f32>(pos, 0.0, 1.0);
}

fn s_curve_activation(x: f32)->f32{
  if x < 0.0 {
    return 0.0;
  }
  if x > 1.0 {
    return 1.0;
  }
  return x * x * (3.0 - 2.0 * x);
}

@fragment
fn main_fs(
  @builtin(position) pos : vec4<f32>
) -> @location(0) vec4<f32> {
  
  let x = u32( pos.x );
  let y = u32( pos.y );

  let index = vec2<u32>(x, y);

  var chemo_value = textureLoad(chemo_in, index, 0);

  // This transform makes it appear less grey/washed out
  let v = s_curve_activation( chemo_value.x );
  chemo_value = vec4<f32>( v, v, v, 1.0);

  // Apply the foreground/background colours
  chemo_value = mix(params.background_colour, params.foreground_colour, chemo_value.r);
  
  var control_value = textureLoad(control_texture, index, 0);

  // mix in the control texture
  var texture_value = mix(chemo_value, control_value, params.control_alpha);

  return texture_value;
}
