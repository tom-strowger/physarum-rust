struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
};

struct SimParams {
  sense_angle : f32,
  sense_offset : f32,
  step : f32,
  rotate_angle : f32,
  max_chemo : f32,
  deposit_chemo : f32,
  decay_chemo : f32,
  sim_width : f32,
  sim_height : f32,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var chemo_in : texture_2d<f32>;
@group(0) @binding(2) var chemo_out: texture_storage_2d<rgba8unorm, write>;

fn 
blur9( texture: texture_2d<f32>, input_uv: vec2<u32>, direction: vec2<i32>, resolution: vec2<f32>) -> vec3<f32> {
  let uv = vec2<i32>( input_uv );

  var color = vec3<f32>(0.0);
  // let off1 = vec2<f32>(1.3846153846) * direction;
  // let off2 = vec2<f32>(3.2307692308) * direction;
  let off1 = vec2<i32>(1, 1) * direction;
  let off2 = vec2<i32>(3, 3) * direction;
  color += textureLoad(texture, uv, 0).rgb * 0.2270270270;
  color += textureLoad(texture, uv + (off1), 0).rgb * 0.3162162162;
  color += textureLoad(texture, uv - (off1), 0).rgb * 0.3162162162;
  color += textureLoad(texture, uv + (off2), 0).rgb * 0.0702702703;
  color += textureLoad(texture, uv - (off2), 0).rgb * 0.0702702703;
  return color;
}

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute
@workgroup_size(8,8)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  
  // Write to chemo_out
  let x = global_invocation_id.x;
  let y = global_invocation_id.y;
  let z = global_invocation_id.z;
  let index = vec2<u32>(x, y);

  // var chemo = textureLoad(chemo_in, index, 0);

  // textureStore(chemo_out, index, vec4<f32>(chemo.r, chemo.g, chemo.b, 1.0));


  // Blur calculations
  var color = vec3<f32>(0.0, 0.0, 0.0);
  let resolution = vec2<f32>( 1.0, 1.0 );

  color += blur9(chemo_in, index, vec2<i32>(1,0), resolution);
  color += blur9(chemo_in, index, vec2<i32>(0,1), resolution);
  color /= vec3<f32>( 2.0, 2.0, 2.0 );

  // decay
  color *= vec3<f32>( 
    1.0 - params.decay_chemo,
    1.0 - params.decay_chemo,
    1.0 - params.decay_chemo );

  if color.r < 0.001 {
    color = vec3<f32>( 0.0, 0.0, 0.0 );
  }

  // let color = textureLoad( chemo_in, index, 0 ).rgb;
  textureStore(chemo_out, index, vec4<f32>(color, 1.0));

}
