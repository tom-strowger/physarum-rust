struct Particle {
  pos : vec2<f32>,
  vel : vec2<f32>,
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
@group(0) @binding(1) var new_dots : texture_2d<f32>;
@group(0) @binding(2) var chemo_in : texture_2d<f32>;
@group(0) @binding(3) var chemo_out: texture_storage_2d<rgba8unorm, write>;

// @group(0) @binding(1) var<storage, read> new_dots : array<Particle>;
// @group(0) @binding(2) var<storage, read> chemo_in : array<Particle>;
// @group(0) @binding(3) var<storage, read_write> chemo_out : array<Particle>;

// https://github.com/austinEng/Project6-Vulkan-Flocking/blob/master/data/shaders/computeparticles/particle.comp
@compute
@workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id: vec3<u32>) {
  // let total = arrayLength(&chemo_in);
  // let index = global_invocation_id.x;
  // if (index >= total) {
  //   return;
  // }

  // // Write back
  // chemo_out[index] = chemo_in[index];
}
