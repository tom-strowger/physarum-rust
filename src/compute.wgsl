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

// @todo This could be improved 
fn rand(co: vec2<f32>)->f32{
    return fract(sin(dot(co, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// @todo Improve sampling 
fn sample_texture(tex: texture_2d<f32>, pos: vec2<f32>)->vec4<f32>{
  let tex_size = textureDimensions(tex);
  let texel_size = vec2<f32>(1.0) / vec2<f32>(tex_size);
  // It seemed like the sampling was off so I added the -0.5, -0.5 which fixed it.
  // There is probably a bug somewhere
  let texel_pos = pos * texel_size -  vec2(0.5, 0.5 );
  return textureSampleLevel(tex, texture_sampler, texel_pos, 0.0);
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
  let sense_a = sample_texture( chemo_texture, pos_a).x;

  // Sense left
  let pos_l = agent.pos * sim_size + dir_l * params.sense_offset;
  let sense_l = sample_texture( chemo_texture, pos_l).x;

  // Sense right
  let pos_r = agent.pos * sim_size  + dir_r * params.sense_offset;
  let sense_r = sample_texture( chemo_texture, pos_r).x;

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
