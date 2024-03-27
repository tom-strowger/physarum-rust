struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
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
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var chemo_in : texture_2d<f32>;
@group(0) @binding(2) var new_dots : texture_2d<f32>;
@group(0) @binding(3) var chemo_out: texture_storage_2d<rgba16float, write>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute
@workgroup_size(8,8)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  
  // Write to chemo_out
  let x = global_invocation_id.x;
  let y = global_invocation_id.y;
  let z = global_invocation_id.z;

  let index = vec2<u32>(x, y);

  let deposit_amount = params.deposit_chemo / params.max_chemo;

  var chemo = textureLoad(chemo_in, index, 0).rgb;
  chemo += textureLoad(new_dots, index, 0).rgb * vec3<f32>( deposit_amount, deposit_amount, deposit_amount );
  chemo = min( chemo, vec3<f32>( 65504.0, 65504.0, 65504.0 ) );

  textureStore(chemo_out, index, vec4<f32>(chemo.r, chemo.g, chemo.b, 1.0));
}
