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
@group(0) @binding(3) var texture_sampler: sampler;


fn 
sample_texture(tex: texture_2d<f32>, pos: vec2<f32>)->vec4<f32>{
  let tex_size = textureDimensions(tex);
  let texel_size = vec2<f32>(1.0) / vec2<f32>(tex_size);

  // the 0.5 offset here is required to compensate for a sort of floor() that seems to happen
  // inside textureSampleLevel
  let texel_pos = (pos + vec2<f32>(0.5, 0.5)) * texel_size;
  return textureSampleLevel(tex, texture_sampler, texel_pos, 0.0);
}

fn 
blur9( texture: texture_2d<f32>, uv: vec2<f32>, direction: vec2<f32>) -> vec3<f32> {
  var color = vec4<f32>(0.0);
  let off1 = vec2<f32>(1.3846153846, 1.3846153846) * direction;
  let off2 = vec2<f32>(3.2307692308, 3.2307692308) * direction;

  let tex_size = vec2<f32>( textureDimensions(texture) );
  let pos = uv;

  // let off1 = vec2<i32>(1, 1) * direction;
  // let off2 = vec2<i32>(3, 3) * direction;
  color += sample_texture(texture, pos) * 0.2270270270;
  color += sample_texture(texture, pos + (off1)) * 0.3162162162;
  color += sample_texture(texture, pos - (off1)) * 0.3162162162;
  // color += sample_texture(texture, pos) * 0.3162162162;
  color += sample_texture(texture, pos + (off2)) * 0.0702702703;
  color += sample_texture(texture, pos - (off2)) * 0.0702702703;
  // color += sample_texture(texture, pos) * 0.0702702703;
  return color.rgb;
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

  color += blur9(chemo_in, vec2<f32>(index), vec2<f32>(1.0,0.0));
  color += blur9(chemo_in, vec2<f32>(index), vec2<f32>(0.0,1.0));
  color /= vec3<f32>( 2.0, 2.0, 2.0 );

  // decay
  color *= vec3<f32>( 
    1.0 - params.decay_chemo,
    1.0 - params.decay_chemo,
    1.0 - params.decay_chemo );

  if color.r < 0.005 {
    color = vec3<f32>( 0.0, 0.0, 0.0 );
  }

  // let color = textureLoad( chemo_in, index, 0 ).rgb;
  textureStore(chemo_out, index, vec4<f32>(color, 1.0));

}
