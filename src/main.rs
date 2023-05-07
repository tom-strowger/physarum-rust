// Flocking boids example with gpu compute update pass
// adapted from https://github.com/austinEng/webgpu-samples/blob/master/src/examples/computeBoids.ts

use nanorand::{Rng, WyRand};
use std::{borrow::Cow, mem};
use wgpu::util::DeviceExt;

#[path = "./framework.rs"]
mod framework;

// number of boid particles to simulate

const NUM_PARTICLES: u32 = 1 << 20;

// number of single-particle calculations (invocations) in each gpu work group

const PARTICLES_PER_GROUP: u32 = 64;

/// Example struct holds references to wgpu resources and frame persistent data
struct Example {
    draw_positions_bind_groups: Vec<wgpu::BindGroup>,
    deposit_bind_group: wgpu::BindGroup,
    diffuse_bind_group: wgpu::BindGroup,
    update_bind_groups: Vec<wgpu::BindGroup>,  // swapped each frame
    render_bind_group: wgpu::BindGroup,

    particle_buffers: Vec<wgpu::Buffer>,
    vertices_buffer: wgpu::Buffer,
    new_dots_texture: wgpu::Texture,

    new_dots_pipeline: wgpu::RenderPipeline,
    deposit_pipeline: wgpu::ComputePipeline,
    diffuse_pipeline: wgpu::ComputePipeline,
    compute_pipeline: wgpu::ComputePipeline,
    
    render_pipeline: wgpu::RenderPipeline,
    frame_num: usize,

    width: u32,
    height: u32,

    sim_param_data: Vec<f32>,
}

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
        _queue: &wgpu::Queue,
    ) -> Self {
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

        let width = 1280;
        let height = 800;

        // buffer for simulation parameters uniform

        let mut sim_param_data = [
            15.0, // sense angle
            4.0,     // sense offset
            1.0,   // step
            35.0,   // rotate angle
            5.0,     // max chemo
            1.0,    // deposit chemo
            0.12,    // decay chemo
            1280.0 * 2.0,   // width
            800.0 * 2.0,    // height
        ]
        .to_vec();
        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            contents: bytemuck::cast_slice(&sim_param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Enough space must be allocated to account for scaling
        let screen_to_texture_ratio = 2;

        let chemo_texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: width * screen_to_texture_ratio,
                height: height * screen_to_texture_ratio,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | 
                wgpu::TextureUsages::RENDER_ATTACHMENT | 
                wgpu::TextureUsages::STORAGE_BINDING,
            label: None,
            view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
        };

        let mut chemo_textures = Vec::new();
        for _i in 0..2 {
            chemo_textures.push( device.create_texture(&chemo_texture_descriptor))
        }

        let new_dots_texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: width * screen_to_texture_ratio,
                height: height * screen_to_texture_ratio,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | 
                wgpu::TextureUsages::RENDER_ATTACHMENT | 
                wgpu::TextureUsages::COPY_DST,
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
                                (sim_param_data.len() * mem::size_of::<f32>()) as _,
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
                            min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
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
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(
                                (sim_param_data.len() * mem::size_of::<f32>()) as _,
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
                                (sim_param_data.len() * mem::size_of::<f32>()) as _,
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
                                (sim_param_data.len() * mem::size_of::<f32>()) as _,
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
                                (sim_param_data.len() * mem::size_of::<f32>()) as _,
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
                            min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
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
                            min_binding_size: wgpu::BufferSize::new((NUM_PARTICLES * 16) as _),
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

        let vertex_buffer_data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0,
                                            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ];
        let vertices_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::bytes_of(&vertex_buffer_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        // buffer for all particles data of type [(posx,posy,velx,vely),...]

        let mut initial_particle_data = vec![0.0f32; (4 * NUM_PARTICLES) as usize];
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

        let mut render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_bind_group_layout,
            entries: &[
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
            vertices_buffer,
            new_dots_texture,

            new_dots_pipeline,
            deposit_pipeline,
            diffuse_pipeline,
            compute_pipeline,

            render_pipeline,
            frame_num: 0,
            width: 100,
            height: 100,
            sim_param_data,
        }
    }

    /// update is called for any WindowEvent not handled by the framework
    fn update(&mut self, _event: winit::event::WindowEvent) {
        //empty
    }

    /// resize is called on WindowEvent::Resized events
    fn resize(
        &mut self,
        sc_desc: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let w = sc_desc.width;
        let h = sc_desc.height;

        self.width = w;
        self.height = h;


        let mut new_data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0,
                                     -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ];

        for vertex in new_data.chunks_mut(2) {
            vertex[0] /= w as f32;
            vertex[1] /= h as f32;
        }

        queue.write_buffer(
            &self.vertices_buffer,
            0, // Offset in the buffer to start writing to
            bytemuck::bytes_of(&new_data)
        );
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
                rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));

                rpass.set_bind_group(0, &self.draw_positions_bind_groups[self.frame_num % 2], &[]);
                
                rpass.draw(0..6, 0..NUM_PARTICLES);

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
                cpass.dispatch_workgroups(NUM_PARTICLES/64, 1, 1);
            }
            command_encoder.pop_debug_group();

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
                    load: wgpu::LoadOp::Load,
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
                rpass.set_vertex_buffer(1, self.vertices_buffer.slice(..));

                rpass.set_bind_group(0, &self.render_bind_group, &[]);
                
                rpass.draw(0..6, 0..1);
            }
            
            command_encoder.pop_debug_group();

        }

        // update frame count
        self.frame_num += 1;

        // done
        queue.submit(Some(command_encoder.finish()));
    }
}

/// run example
fn main() {
    framework::run::<Example>("boids");
}
