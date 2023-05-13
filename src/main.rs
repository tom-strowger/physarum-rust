// Flocking boids example with gpu compute update pass
// adapted from https://github.com/austinEng/webgpu-samples/blob/master/src/examples/computeBoids.ts

use nanorand::{Rng, WyRand};
use std::{borrow::Cow, mem};
use wgpu::{util::DeviceExt, Buffer};
use winit::{
    event::{self, WindowEvent}};
use image::{ImageBuffer, Rgba};
use chrono::Utc;
use bytemuck::{Pod, Zeroable, NoUninit};
use serde::{Serialize, Deserialize};
use std::io::Write;

#[path = "./framework.rs"]
mod framework;

// define a global frame rate limit
const FRAME_RATE_LIMIT: Option<u32> = Some(120);

/// Example struct holds references to wgpu resources and frame persistent data
struct Example {
    draw_positions_bind_groups: Vec<wgpu::BindGroup>,
    deposit_bind_group: wgpu::BindGroup,
    diffuse_bind_group: wgpu::BindGroup,
    update_bind_groups: Vec<wgpu::BindGroup>,  // swapped each frame
    render_bind_group: wgpu::BindGroup,

    particle_buffers: Vec<wgpu::Buffer>,
    new_dots_vertices_buffer: wgpu::Buffer,
    new_dots_texture: wgpu::Texture,
    chemo_textures: Vec<wgpu::Texture>,
    control_texture: wgpu::Texture,
    whole_view_vertices_buffer: wgpu::Buffer,

    new_dots_pipeline: wgpu::RenderPipeline,
    deposit_pipeline: wgpu::ComputePipeline,
    diffuse_pipeline: wgpu::ComputePipeline,
    compute_pipeline: wgpu::ComputePipeline,
    
    render_pipeline: wgpu::RenderPipeline,
    frame_num: usize,

    width: u32,
    height: u32,

    running: bool,
    save: bool,
    dump: bool,

    // next time to render
    next_render_time: std::time::Instant,

    sim_param_data: SimulationParameters,
    sim_param_buffer: wgpu::Buffer,
    sim_param_data_dirty: bool,
}
// this is Pod
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
struct SimulationParameters
{
    sense_angle: f32,
    sense_offset: f32,
    step: f32,
    rotate_angle: f32,
    max_chemo: f32,
    deposit_chemo: f32,
    decay_chemo: f32,
    width: u32,
    height: u32,
    control_alpha: f32,
    num_particles: u32,
}

impl SimulationParameters
{
    fn serialize(& self) -> String
    {
        return format!("sa{:.2}_so{:.2}_step{:.2}_ra{:.2}_decay{:.2}_N{}",
            self.sense_angle,
            self.sense_offset,
            self.step,
            self.rotate_angle,
            self.decay_chemo,
            self.num_particles
        ).to_string();
    }
}

unsafe impl Zeroable for SimulationParameters {}
unsafe impl Pod for SimulationParameters {}
// unsafe impl NoUninit for SimulationParameters {}

impl framework::Example for Example {
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    /// constructs initial instance of Example struct
    fn init(
        config: &wgpu::SurfaceConfiguration,
        _adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Self {
        
        
        let width = 1280;
        let height = 800;

        let dpi_factor = config.width as f32 / width as f32;


        let width = 1280;
        let height = 800;

        let dpi_factor = config.width as f32 / width as f32;

        let deposit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("deposit.wgsl"))),
        });
        let diffuse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("diffuse.wgsl"))),
        });
        
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
        });
        let new_dots_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw_new_dots.wgsl"))),
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw.wgsl"))),
        });

        // buffer for simulation parameters uniform

        // a pleasing textural set
        // let sim_param_data = SimulationParameters
        // {
        //     sense_angle: 15.0,
        //     sense_offset: 3.0,
        //     step: 3.0,
        //     rotate_angle: 5.0,
        //     max_chemo: 5.0,
        //     deposit_chemo: 1.0,
        //     decay_chemo: 0.10,
        //     width: config.width,
        //     height: config.height,
        //     num_particles: 1 << 20,
        // };
        
        let sim_param_data = SimulationParameters
        {
            sense_angle: 15.0,
            sense_offset: 4.0,
            step: 3.5,
            rotate_angle: 18.0,
            max_chemo: 5.0,
            deposit_chemo: 1.0,
            decay_chemo: 0.10,
            width: config.width,
            height: config.height,
            control_alpha: 0.2,
            num_particles: 1 << 20,
        };
        
        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            contents: bytemuck::bytes_of(&sim_param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let chemo_texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2, 
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | 
                wgpu::TextureUsages::RENDER_ATTACHMENT | 
                wgpu::TextureUsages::STORAGE_BINDING |
                wgpu::TextureUsages::COPY_SRC,
            label: None,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        };

        let mut chemo_textures = Vec::new();
        for _i in 0..2 {
            chemo_textures.push( device.create_texture(&chemo_texture_descriptor))
        }
        
        let control_texture;

        // check if there is an image to load to the control texture
        let image_path = "control.png";
        
        if std::path::Path::new(image_path).exists() {
            let img = image::open(image_path).unwrap().to_rgba8();
            let img_dimensions = img.dimensions();
            let img_data = img.into_raw();

            // scale the texture data to fit the texture
            let mut scaled_img_data = Vec::new();
            scaled_img_data.reserve((config.width * config.height * 4) as usize);
            for y in 0..config.height {
                for x in 0..config.width {
                    let scaled_x = (x as f32 / config.width as f32 * img_dimensions.0 as f32) as u32;
                    let scaled_y = (y as f32 / config.height as f32 * img_dimensions.1 as f32) as u32;
                    let scaled_index = (scaled_y * img_dimensions.0 + scaled_x) as usize * 4;
                    scaled_img_data.push(img_data[scaled_index]);
                    scaled_img_data.push(img_data[scaled_index + 1]);
                    scaled_img_data.push(img_data[scaled_index + 2]);
                    scaled_img_data.push(img_data[scaled_index + 3]);
                }
            }            
            control_texture = device.create_texture_with_data(queue, &chemo_texture_descriptor, &scaled_img_data );
        }
        else {
            // If there isn't an image on disk, create an empty control texture
            control_texture = device.create_texture(&chemo_texture_descriptor);
        }

        let new_dots_texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | 
                wgpu::TextureUsages::RENDER_ATTACHMENT | 
                wgpu::TextureUsages::COPY_SRC,
            label: None,
            view_formats: &[wgpu::TextureFormat::Bgra8Unorm],
        };
        
        let new_dots_texture = device.create_texture(&new_dots_texture_descriptor);


        // create compute bind layout group and compute pipeline layout
        let draw_position_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[

                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<SimulationParameters>() as _,
                            ),
                        },
                        count: None,
                    },

                    // input agents
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((sim_param_data.num_particles * 16) as _),
                        },
                        count: None,
                    }
                ],
                label: None,
            });

    
        let render_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[

                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX |
                                    wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<SimulationParameters>() as _,
                            ),
                        },
                        count: None,
                    },

                    // chemo to draw
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: false,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },
                    
                    // conmtrol texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: false,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        let deposit_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[

                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<SimulationParameters>() as _,
                            ),
                        },
                        count: None,
                    },

                    // input chemo texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: false,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },

                    // input new dots
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: false,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },

                    // output chemo texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
                label: None,
            });

        // create a texture sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 10,
            border_color: None,
        });

        let diffuse_bind_group_layout = 
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[

                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<SimulationParameters>() as _,
                            ),
                        },
                        count: None,
                    },

                    // input chemo texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },

                    // output chemo texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },

                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering
                        ),
                        count: None,
                    },
                ],
                label: None,
            });


        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[

                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                mem::size_of::<SimulationParameters>() as _,
                            ),
                        },
                        count: None,
                    },

                    // input agents
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((sim_param_data.num_particles * 16) as _),
                        },
                        count: None,
                    },

                    // input chemo
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },

                    // output agents
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new((sim_param_data.num_particles * 16) as _),
                        },
                        count: None,
                    },

                    // sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering
                        ),
                        count: None,
                    },
                    
                    // input control
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float{
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false
                        },
                        count: None,
                    },
                ],
                label: None,
            });


        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });


        let deposit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("deposit"),
                bind_group_layouts: &[&deposit_bind_group_layout],
                push_constant_ranges: &[],
            });

        let diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("diffuse"),
                bind_group_layouts: &[&diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });

        // create render pipeline with empty bind group layout


        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
            });
        
        let new_dots_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("new dots"),
                bind_group_layouts: &[&draw_position_bind_group_layout],
                push_constant_ranges: &[],
            });

        let new_dots_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&new_dots_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &new_dots_shader,
                entry_point: "main_vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 4 * 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &new_dots_shader,
                entry_point: "main_fs",
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: "main_vs",
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 4 * 4,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32],
                    },
                    wgpu::VertexBufferLayout {
                        array_stride: 2 * 4,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![2 => Float32x2],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: "main_fs",
                targets: &[Some(config.view_formats[0].into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        // create compute pipeline

        let deposit_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Deposit pipeline"),
            layout: Some(&deposit_pipeline_layout),
            module: &deposit_shader,
            entry_point: "main",
        });

        let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse pipeline"),
            layout: Some(&diffuse_pipeline_layout),
            module: &diffuse_shader,
            entry_point: "main",
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        // buffer for the 2x triangle vertices of each instance, together making a square

        let mut vertex_buffer_data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0,
                                            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ];

        let whole_view_vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        for vertex in vertex_buffer_data.chunks_mut(2) {
            vertex[0] /= config.width as f32;
            vertex[1] /= config.height as f32;
        }

        let new_dots_vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // buffer for all particles data of type [(posx,posy,velx,vely),...]

        let mut initial_particle_data = vec![0.0f32; (4 * sim_param_data.num_particles) as usize];
        let mut rng = WyRand::new_seed(42);
        let mut unif = || rng.generate::<f32>(); // Generate a num (0, 1)
        for particle_instance_chunk in initial_particle_data.chunks_mut(4) {
            particle_instance_chunk[0] = unif(); // posx
            particle_instance_chunk[1] = unif(); // posy
            particle_instance_chunk[2] = unif(); // heading
            particle_instance_chunk[3] = 0.0f32;  // padding
        }

        // creates two buffers of particle data each of size NUM_PARTICLES
        // the two buffers alternate as dst and src for each frame

        let mut particle_buffers = Vec::<wgpu::Buffer>::new();
        let mut particle_bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            particle_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Particle Buffer {i}")),
                    contents: bytemuck::cast_slice(&initial_particle_data),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                }),
            );
        }

        // create two bind groups, one for each buffer as the src
        // where the alternate buffer is used as the dst
        let mut draw_positions_bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            draw_positions_bind_groups.push( device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &draw_position_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_param_buffer.as_entire_binding(),
                    },

                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[i].as_entire_binding(),
                    },
                ],
                label: None,
            }));
        }

        let mut deposit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &deposit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_param_buffer.as_entire_binding(),
                },

                // chemo in
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &chemo_textures[0].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // new dots
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &new_dots_texture.create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // chemo out
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &chemo_textures[1].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },
            ],
            label: None,
        });


        let mut diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &diffuse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_param_buffer.as_entire_binding(),
                },

                // chemo in
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &chemo_textures[1].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // chemo out
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &chemo_textures[0].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // sampler
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(
                        &sampler
                    )
                },
            ],
            label: None,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_bind_group_layout,
            entries: &[

                // params
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_param_buffer.as_entire_binding(),
                },

                // chemo
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &chemo_textures[0].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },
                
                // control
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &control_texture.create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },
            ],
            label: None,
        });


        for i in 0..2 {
            particle_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &compute_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: sim_param_buffer.as_entire_binding(),
                    },
                    // agents in
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: particle_buffers[i].as_entire_binding(),
                    },

                    // chemo in
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &chemo_textures[0].create_view(
                                &wgpu::TextureViewDescriptor::default())
                        )
                    },

                    // agents out
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: particle_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
                    },

                    // sampler
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(
                            &sampler
                        )
                    },

                    // control
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(
                            &control_texture.create_view(
                                &wgpu::TextureViewDescriptor::default())
                        )
                    },
                ],
                label: None,
            }));
        }

        // returns Example struct and No encoder commands
        Example {
            draw_positions_bind_groups,
            deposit_bind_group,
            diffuse_bind_group,
            update_bind_groups: particle_bind_groups,
            render_bind_group,

            particle_buffers,
            new_dots_vertices_buffer,
            new_dots_texture,
            chemo_textures,
            control_texture,
            whole_view_vertices_buffer,

            new_dots_pipeline,
            deposit_pipeline,
            diffuse_pipeline,
            compute_pipeline,

            render_pipeline,
            frame_num: 0,
            width: config.width,
            height: config.height,

            running: true,
            save: false,
            dump: false,

            next_render_time: std::time::Instant::now(),

            sim_param_data,
            sim_param_buffer,
            sim_param_data_dirty: false,
        }
    }

    fn write_texture_to_image(&self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        texture: &wgpu::Texture, 
        file_name: &str)
    {
        
        let buffer_data = vec![0.0f32; (4 * self.width * self.height) as usize];
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&"buffer data"),
            contents: bytemuck::cast_slice(&buffer_data),
            usage: wgpu::BufferUsages::MAP_READ
                | wgpu::BufferUsages::COPY_DST,
        });


        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::ImageCopyBuffer{
                buffer: &buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * self.width),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(encoder.finish()));

        let buffer_slice = buffer.slice(..);

        // map the buffer, wait until the callback is called
        
        let width = self.width;
        let height = self.height;

        buffer_slice.map_async(wgpu::MapMode::Read,|slice| {
            if let Err(_) = slice {
                panic!("failed to map buffer");
            }
         });
        device.poll(wgpu::Maintain::Wait);

        let data = buffer_slice.get_mapped_range();
        let buf = data.to_vec();
        let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(width, height, buf).unwrap();

        img.save(file_name).unwrap();
        drop(data);

        buffer.unmap();
    }

    /// update is called for any WindowEvent not handled by the framework
    fn update(&mut self, event: winit::event::WindowEvent) {
        //empty

        match event
        {
            WindowEvent::KeyboardInput {
                input:
                    event::KeyboardInput {
                        virtual_keycode: Some(event::VirtualKeyCode::Space),
                        state: event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Toggle pause
                self.running = !self.running;
            },
            WindowEvent::KeyboardInput {
                input:
                    event::KeyboardInput {
                        virtual_keycode: Some(event::VirtualKeyCode::S),
                        state: event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Toggle pause
                self.save = true;
            },
            WindowEvent::KeyboardInput {
                input:
                    event::KeyboardInput {
                        virtual_keycode: Some(event::VirtualKeyCode::D),
                        state: event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Toggle pause
                self.dump = true;
            },
            WindowEvent::KeyboardInput {
                input:
                    event::KeyboardInput {
                        virtual_keycode: Some(event::VirtualKeyCode::C),
                        state: event::ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                // Adjust the control alpha
                if self.sim_param_data.control_alpha == 0.0 {
                    self.sim_param_data.control_alpha = 0.2;
                }
                else if self.sim_param_data.control_alpha == 0.2 {
                    self.sim_param_data.control_alpha = 1.0;
                }
                else {
                    self.sim_param_data.control_alpha = 0.0;
                }

                self.sim_param_data_dirty = true;
            },
            _ => {}
        }
    }

    /// resize is called on WindowEvent::Resized events
    fn resize(
        &mut self,
        sc_desc: &wgpu::SurfaceConfiguration,
        _device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) {
        let w = sc_desc.width;
        let h = sc_desc.height;

        self.width = w;
        self.height = h;
    }

    /// render is called each frame, dispatching compute groups proportional
    ///   a TriangleList draw call for all NUM_PARTICLES at 3 vertices each
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        
        // get command encoder
        let mut command_encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if self.sim_param_data_dirty
        {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&"buffer data"),
                contents: bytemuck::bytes_of(&self.sim_param_data),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            encoder.copy_buffer_to_buffer(
                &buffer,
                0,
                &self.sim_param_buffer,
                0,
                std::mem::size_of::<SimulationParameters>() as wgpu::BufferAddress,
            );
            queue.submit(Some(encoder.finish()));
        }

        let time_to_render = FRAME_RATE_LIMIT.is_none() || std::time::Instant::now() >= self.next_render_time;

        if time_to_render
        {
            if let(Some(frame_rate_limit)) = FRAME_RATE_LIMIT {
                self.next_render_time += std::time::Duration::from_millis((1000.0 / frame_rate_limit as f32) as u64);
            }

            if self.running
            {
            
                // render to new_dots_texture
                {
                    // create a texture view from new_dots_texture
                    let new_dots_texture_view = self.new_dots_texture.create_view(&wgpu::TextureViewDescriptor::default());

                    // create render pass descriptor and its color attachments
                    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                        view: &new_dots_texture_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // Not clearing here in order to test wgpu's zero texture initialization on a surface texture.
                            // Users should avoid loading uninitialized memory since this can cause additional overhead.
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: true,
                        },
                    })];
                    let new_dots_render_pass_descriptor = wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &color_attachments,
                        depth_stencil_attachment: None,
                    };

                    command_encoder.push_debug_group("render dots");
                    {
                        // render pass
                        let mut rpass = command_encoder.begin_render_pass(&new_dots_render_pass_descriptor);
                        rpass.set_pipeline(&self.new_dots_pipeline);
                        
                        // render dst particles
                        rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
                        // the three instance-local vertices
                        rpass.set_vertex_buffer(1, self.new_dots_vertices_buffer.slice(..));

                        rpass.set_bind_group(0, &self.draw_positions_bind_groups[self.frame_num % 2], &[]);
                        
                        rpass.draw(0..6, 0..self.sim_param_data.num_particles);

                    }
                    command_encoder.pop_debug_group();
                    

                    let work_group_size = 8;

                    command_encoder.push_debug_group("deposit chemo");
                    {
                        // compute pass
                        let mut cpass =
                            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                        cpass.set_pipeline(&self.deposit_pipeline);
                        cpass.set_bind_group(0, &self.deposit_bind_group, &[]);
                        cpass.dispatch_workgroups(self.width/work_group_size, self.height/work_group_size, 1);
                    }
                    command_encoder.pop_debug_group();


                    command_encoder.push_debug_group("diffuse chemo");
                    {
                        // compute pass
                        let mut cpass =
                            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                        cpass.set_pipeline(&self.diffuse_pipeline);
                        cpass.set_bind_group(0, &self.diffuse_bind_group, &[]);
                        cpass.dispatch_workgroups(self.width/work_group_size, self.height/work_group_size, 1);
                    }
                    command_encoder.pop_debug_group();


                    command_encoder.push_debug_group("update agent positions");
                    {
                        // compute pass
                        let mut cpass =
                            command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                        cpass.set_pipeline(&self.compute_pipeline);
                        cpass.set_bind_group(0, &self.update_bind_groups[self.frame_num % 2], &[]);
                        cpass.dispatch_workgroups(self.sim_param_data.num_particles/64, 1, 1);
                    }
                    command_encoder.pop_debug_group();

                }
                    
            }
        }

        // render to view
        {

            // create render pass descriptor and its color attachments
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: view,
                resolve_target: None,
                ops: wgpu::Operations {
                    // Not clearing here in order to test wgpu's zero texture initialization on a surface texture.
                    // Users should avoid loading uninitialized memory since this can cause additional overhead.
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: true,
                },
            })];
            let render_pass_descriptor = wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
            };

            command_encoder.push_debug_group("render view");
            {
                // render pass
                let mut rpass = command_encoder.begin_render_pass(&render_pass_descriptor);
                rpass.set_pipeline(&self.render_pipeline);
                
                rpass.set_vertex_buffer(0, self.particle_buffers[(self.frame_num + 1) % 2].slice(..));
                rpass.set_vertex_buffer(1, self.whole_view_vertices_buffer.slice(..));

                rpass.set_bind_group(0, &self.render_bind_group, &[]);
                
                rpass.draw(0..6, 0..1);
            }
            
            command_encoder.pop_debug_group();

        }

        // save the chemo texture to an image
        if self.save
        {
            self.save = false;

            let now = Utc::now();

            self.write_texture_to_image(
                device, 
                queue, 
                &self.chemo_textures[0], 
                format!(
                    "chemo_{}_{}.png",
                    now.format("%Y%m%d_%H%M"), 
                    self.sim_param_data.serialize()).as_str() );
        }

        if self.dump
        {
            self.dump = false;

            // get the date time
            let now = Utc::now();

            // make a directory for the dump
            let dump_dir = format!("dump_{}", now.format("%Y%m%d_%H%M"));
            std::fs::create_dir(dump_dir.as_str()).unwrap();

            // dump the parameters as json
            let param_json = serde_json::to_string_pretty(&self.sim_param_data).unwrap();
            let mut param_file = std::fs::File::create(format!("{}/params.json", dump_dir).as_str()).unwrap();
            param_file.write_all(param_json.as_bytes()).unwrap();


            // write the textures as images
            self.write_texture_to_image(
                device, 
                queue, 
                &self.chemo_textures[0], 
                format!("{}/chemo_0.png", dump_dir).as_str() );
            self.write_texture_to_image(
                device, 
                queue, 
                &self.chemo_textures[1], 
                format!("{}/chemo_1.png", dump_dir).as_str() );
            self.write_texture_to_image(
                device, 
                queue, 
                &&self.control_texture, 
                format!("{}/control.png", dump_dir).as_str() );
            self.write_texture_to_image(
                device, 
                queue, 
                &&self.new_dots_texture, 
                format!("{}/new_dots.png", dump_dir).as_str() );
        }

        // update frame count
        self.frame_num += 1;

        // done
        queue.submit(Some(command_encoder.finish()));
    }
}

/// run example
fn main() {
    framework::run::<Example>("Physarum", (1920, 1280));
}
