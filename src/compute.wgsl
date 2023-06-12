struct Particle {
  pos : vec2<f32>,
  heading : f32,
  padding : f32,
};

struct SimParams {
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
@group(0) @binding(1) var<storage, read> agents_in : array<Particle>;
@group(0) @binding(2) var chemo_texture : texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> agents_out : array<Particle>;
@group(0) @binding(4) var texture_sampler: sampler;
@group(0) @binding(5) var control_texture : texture_2d<f32>;

// @todo This could be improved 
fn rand(co: vec2<f32>)->f32{
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}


fn sample_texture(tex: texture_2d<f32>, pos: vec2<f32>)->vec4<f32>{
  let tex_size = textureDimensions(tex);
  let texel_size = vec2<f32>(1.0) / vec2<f32>(tex_size);
  // It seemed like the sampling was off so I added the -0.5, -0.5 which fixed it.
  // There is probably a bug somewhere

  var texel_pos = pos * texel_size;
  return textureSampleLevel(tex, texture_sampler, texel_pos, 0.0);

  // var texel_pos = pos * texel_size;
  // if( texel_pos.y < 0.9 )
  // {
  //   return textureSampleLevel(tex, texture_sampler, texel_pos, 0.0);
  // }
  // else
  // {
  //   return vec4<f32>(0.0, 0.0, 0.0, 0.0);
  // }
}

fn sample_texture_control(tex: texture_2d<f32>, pos: vec2<f32>)->vec4<f32>{
  let tex_size = textureDimensions(tex);
  let texel_size = vec2<f32>(1.0) / vec2<f32>(tex_size);
  // It seemed like the sampling was off so I added the -0.5, -0.5 which fixed it.
  // There is probably a bug somewhere
  let texel_pos = pos * texel_size;
  return textureSampleLevel(tex, texture_sampler, texel_pos, 0.0);
}

fn sense_at_location(pos: vec2<f32>)->f32{
  let chemo_sample = sample_texture(chemo_texture, pos).r;
  let control_sample = sample_texture_control(control_texture, pos);

  // let control_sample = textureLoad(control_texture, vec2<u32>( pos ), 0);

  // Red repels, blue attracts
  return chemo_sample * ( 1.0 - control_sample.r ) * (1.0 + control_sample.b);
}

@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  let total = arrayLength(&agents_in);
  let index = global_invocation_id.x;
  if (index >= total) {
    return;
  }

  // let sim_size = vec2<f32>(params.sim_width, params.sim_height);
  let sim_size = vec2<f32>(textureDimensions(chemo_texture));

  var agent = agents_in[index];

  let head_a = agent.heading * 2.0 * radians( 180.0 );
  let head_l = head_a - radians( params.sense_angle );
  let head_r = head_a + radians( params.sense_angle );

  let dir_a = vec2<f32>( sin( head_a ), cos( head_a ) );
  let dir_l = vec2<f32>( sin( head_l ), cos( head_l ) );
  let dir_r = vec2<f32>( sin( head_r ), cos( head_r ) );

  // Sense ahead
  // @todo check texture sampling is repeating (not clamping)
  let pos_a = agent.pos * sim_size + dir_a * params.sense_offset;
  let sense_a = sense_at_location( pos_a );

  // Sense left
  let pos_l = agent.pos * sim_size + dir_l * params.sense_offset;
  let sense_l = sense_at_location( pos_l );

  // Sense right
  let pos_r = agent.pos * sim_size  + dir_r * params.sense_offset;
  let sense_r = sense_at_location( pos_r );

  var dir = dir_a;
  var head = head_a;

  if(sense_l > sense_a && sense_a >= sense_r )
  {
      dir = dir_l;
      head = head_l;
  }
  else if( sense_r > sense_a && sense_a >= sense_l )
  {
      dir = dir_r;
      head = head_r;
  }
  else if( sense_r > sense_a && sense_l > sense_a )
  {
      if( rand(pos_a) > 0.5 )
      {
          dir = dir_r;
          head = head_r;
      }
      else
      {
          dir = dir_l;
          head = head_l;
      }
  }

  agent.pos += dir * vec2<f32>( params.step, params.step ) / sim_size;

  if( agent.pos.x > 1. )
  {
      agent.pos.x -= 1.;
  }
  else if( agent.pos.x < 0. )
  {
      agent.pos.x += 1.;
  }

  if( agent.pos.y > 1. )
  {
      agent.pos.y -= 1.;
  }
  else if( agent.pos.y < 0. )
  {
      agent.pos.y += 1.;
  }

  // Instread of the below, there were quite nice results with this bug present
  // agent.b = head / ( 2 * radians( 180 ) );

  // Normalise the heading and take the fractional part (whole rotations don't matter)
  head /= 2.0 * radians( 180.0 );
  agent.heading = fract( head );

  // Write back
  agents_out[index] = agent;
}
