use nanorand::{Rng, WyRand};
use std::{borrow::Cow, mem};
use wgpu::{util::DeviceExt};
use chrono::Utc;
use bytemuck::{Pod, Zeroable};
use serde::{Serialize, Deserialize};
use std::io::Write;

/// Pipeline struct holds references to wgpu resources and frame persistent data

struct PipelineSharedBuffers {
    // Buffers/data used during the pipelines
    agent_buffers: Vec<wgpu::Buffer>,
    new_dots_vertices_buffer: wgpu::Buffer,
    new_dots_texture: wgpu::Texture,
    chemo_textures: Vec<wgpu::Texture>,
    control_texture: wgpu::Texture,

    sim_param_buffer: wgpu::Buffer,
    sim_param_data: SimulationParameters,
}

trait RenderableStage {
    fn render(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        view: &wgpu::TextureView,
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize );
}


trait ExecutableStage {
    fn execute(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize );
}

pub struct Pipeline {
    // Buffers/data used during the pipelines
    shared_buffers: PipelineSharedBuffers,

    // Pipeline stages, each is a render/compute pipeline 
    executable_stages : Vec<Box<dyn ExecutableStage>>,

    render_stage: Box<dyn RenderableStage>,
    
    width: u32,
    height: u32,

    frame_num: usize,

    sim_param_data_dirty: bool,
}

pub struct PipelineConfiguration
{
    agent_buffers: Vec<wgpu::Buffer>,
    new_dots_vertices_buffer: wgpu::Buffer,
    new_dots_texture: wgpu::Texture,
    chemo_textures: Vec<wgpu::Texture>,
    control_texture: wgpu::Texture,
    
    sim_param_data: SimulationParameters,
}

impl PipelineConfiguration {

    pub fn default(
        device: &wgpu::Device,
        config: &wgpu::SurfaceConfiguration,
        queue: &wgpu::Queue,
    ) -> PipelineConfiguration {

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
        //     num_agents: 1 << 20,
        // };
        
        let sim_param_data: SimulationParameters = SimulationParameters
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
            num_agents: 1 << 20,
        };


        // buffer for the 2x triangle vertices of each instance, together making a square

        let mut vertex_buffer_data = [-1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0,
                                            -1.0, -1.0, -1.0, 1.0, 1.0, 1.0 ];

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

        let mut initial_particle_data = vec![0.0f32; (4 * sim_param_data.num_agents) as usize];
        let mut rng = WyRand::new_seed(42);
        let mut unif = || rng.generate::<f32>(); // Generate a num (0, 1)
        for particle_instance_chunk in initial_particle_data.chunks_mut(4) {
            particle_instance_chunk[0] = unif(); // posx
            particle_instance_chunk[1] = unif(); // posy
            particle_instance_chunk[2] = unif(); // heading
            particle_instance_chunk[3] = 0.0f32;  // padding
        }

        // creates two buffers of agent data each of size NUM_AGENTS
        // the two buffers alternate as dst and src for each frame during the agent update compute pass
        let mut agent_buffers = Vec::<wgpu::Buffer>::new();
        for i in 0..2 {
            agent_buffers.push(
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(&format!("Particle Buffer {i}")),
                    contents: bytemuck::cast_slice(&initial_particle_data),
                    usage: wgpu::BufferUsages::VERTEX
                        | wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST,
                }),
            );
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

        PipelineConfiguration
        {
            agent_buffers,
            new_dots_vertices_buffer,
            new_dots_texture,
            chemo_textures,
            control_texture,

            sim_param_data
        }
    }
}

// this is Pod
#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub struct SimulationParameters
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
    num_agents: u32,
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
            self.num_agents
        ).to_string();
    }
}

const LOGICAL_WIDTH : u32 = 1280;
const LOGICAL_HEIGHT : u32 = 800;

unsafe impl Zeroable for SimulationParameters {}
unsafe impl Pod for SimulationParameters {}
// unsafe impl NoUninit for SimulationParameters {}

impl Pipeline {
    pub fn required_limits() -> wgpu::Limits {
        wgpu::Limits::downlevel_defaults()
    }

    pub fn required_downlevel_capabilities() -> wgpu::DownlevelCapabilities {
        wgpu::DownlevelCapabilities {
            flags: wgpu::DownlevelFlags::COMPUTE_SHADERS,
            ..Default::default()
        }
    }

    /// constructs initial instance of Example struct
    pub fn init(
        config: &wgpu::SurfaceConfiguration,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pipeline_configuration: PipelineConfiguration
    ) -> Self {

        let dpi_factor = config.width as f32 / LOGICAL_WIDTH as f32;

        let sim_param_data = pipeline_configuration.sim_param_data;
        let agent_buffers = pipeline_configuration.agent_buffers;
        let new_dots_vertices_buffer = pipeline_configuration.new_dots_vertices_buffer;
        let new_dots_texture = pipeline_configuration.new_dots_texture;
        let chemo_textures = pipeline_configuration.chemo_textures;
        let control_texture = pipeline_configuration.control_texture;
        // let whole_view_vertices_buffer = pipeline_configuration.whole_view_vertices_buffer;
        
        let sim_param_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Simulation Parameter Buffer"),
            contents: bytemuck::bytes_of(&sim_param_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });


        // create a texture sampler
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: wgpu::AddressMode::ClampToEdge,  // temporaririly clamping while debugging.  Will move back to repeating later
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 100.0,
            compare: None,
            anisotropy_clamp: 10,
            border_color: None,
        });

        let shared_buffers = PipelineSharedBuffers {
            agent_buffers,
            new_dots_vertices_buffer,
            new_dots_texture,
            chemo_textures,
            control_texture,

            sim_param_data,
            sim_param_buffer,
        };

        let width = config.width;
        let height = config.height;

        let mut executable_stages : Vec<Box<dyn ExecutableStage>> = Vec::new();

        executable_stages.push(Box::new(NewDotsPipelineStage::create(device, config, &sim_param_data, &shared_buffers)));
        executable_stages.push(Box::new(DepositPipelineStage::create(device, config, &sim_param_data, &shared_buffers, width, height )));
        executable_stages.push(Box::new(DiffusePipelineStage::create(device, config, &sim_param_data, &shared_buffers, width, height, &sampler )));
        executable_stages.push(Box::new(UpdateAgentsPipelineStage::create(device, config, &sim_param_data, &shared_buffers, width, height, &sampler )));
        
        let render_stage = Box::new( RenderPipelineStage::create(device, config, &sim_param_data, &shared_buffers, width, height, &sampler));
        
        // returns Example struct and No encoder commands
        Pipeline {
            shared_buffers,

            executable_stages,
            render_stage,

            width,
            height,

            frame_num: 0,

            sim_param_data_dirty: false,
        }
    }

    pub fn get_control_alpha(
        &self ) -> f32 {
        self.shared_buffers.sim_param_data.control_alpha
    }

    pub fn set_control_alpha(
        &mut self,
        alpha: f32 ) {
        self.shared_buffers.sim_param_data.control_alpha = alpha;
        self.sim_param_data_dirty = true;
    }

    pub fn save_image(&self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        file_name_prefix: &str)
    {
        
        let file_name = format!("{}_{}.png", file_name_prefix, self.shared_buffers.sim_param_data.serialize());
        let texture = &self.shared_buffers.chemo_textures[0];

        self.write_texture_to_image(device, queue, texture, file_name.as_str());

    }

    pub fn dump_debug_textures(&self, 
        device: &wgpu::Device,
        queue: &wgpu::Queue)
    {
        
        // get the date time
        let now = Utc::now();

        // make a directory for the dump
        let dump_dir = format!("dump_{}", now.format("%Y%m%d_%H%M"));
        std::fs::create_dir(dump_dir.as_str()).unwrap();

        // dump the parameters as json
        let param_json = serde_json::to_string_pretty(&self.shared_buffers.sim_param_data).unwrap();
        let mut param_file = std::fs::File::create(format!("{}/params.json", dump_dir).as_str()).unwrap();
        param_file.write_all(param_json.as_bytes()).unwrap();


        // write the textures as images
        self.write_texture_to_image(
            device, 
            queue, 
            &self.shared_buffers.chemo_textures[0], 
            format!("{}/chemo_0.png", dump_dir).as_str() );
        self.write_texture_to_image(
            device, 
            queue, 
            &self.shared_buffers.chemo_textures[1], 
            format!("{}/chemo_1.png", dump_dir).as_str() );
        self.write_texture_to_image(
            device, 
            queue, 
            &&self.shared_buffers.control_texture, 
            format!("{}/control.png", dump_dir).as_str() );
        self.write_texture_to_image(
            device, 
            queue, 
            &&self.shared_buffers.new_dots_texture, 
            format!("{}/new_dots.png", dump_dir).as_str() );
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

    pub fn resize(
        &mut self,
        w: u32,
        h: u32,
    ) {
        self.width = w;
        self.height = h;
    }

    fn do_parameter_update(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue
    )
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        if self.sim_param_data_dirty
        {
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&"buffer data"),
                contents: bytemuck::bytes_of(&self.shared_buffers.sim_param_data),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

            encoder.copy_buffer_to_buffer(
                &buffer,
                0,
                &self.shared_buffers.sim_param_buffer,
                0,
                std::mem::size_of::<SimulationParameters>() as wgpu::BufferAddress,
            );
            queue.submit(Some(encoder.finish()));
            self.sim_param_data_dirty = false;
        }
    }

    /// render is called each frame, dispatching compute groups proportional
    ///   a TriangleList draw call for all NUM_AGENTS at 3 vertices each
    pub fn compute_step(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue
    ) {
        self.do_parameter_update( device, queue);

        // get command encoder
        let mut command_encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // execute the pipeline stages
        for s in &self.executable_stages{
            s.as_ref().execute( &mut command_encoder, &self.shared_buffers, self.frame_num);
        }

        // update frame count
        self.frame_num += 1;

        queue.submit(Some(command_encoder.finish()));
    }

    pub fn render(
        &mut self,
        view: &wgpu::TextureView,
        device: &wgpu::Device,
        queue: &wgpu::Queue
    ) {
        
        self.do_parameter_update( device, queue);

        // get command encoder
        let mut command_encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            
        // render to view
        self.render_stage.as_ref().render( &mut command_encoder, view, &self.shared_buffers, self.frame_num);

        // done
        queue.submit(Some(command_encoder.finish()));
    }


}



struct NewDotsPipelineStage
{
    bind_groups: Vec::<wgpu::BindGroup>,
    pipeline: wgpu::RenderPipeline,
}

impl NewDotsPipelineStage
{
    fn create(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,
        sim_param_data: &SimulationParameters,
        shared_buffers: &PipelineSharedBuffers,
    ) -> NewDotsPipelineStage
    {
        let new_dots_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw_new_dots.wgsl"))),
        });
    
        // create compute bind layout group and compute pipeline layout
        let bind_group_layout = 
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
                            min_binding_size: wgpu::BufferSize::new((sim_param_data.num_agents * 16) as _),
                        },
                        count: None,
                    }
                ],
                label: None,
            });

        // create render pipeline with empty bind group layout        
        let new_dots_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("new dots"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        let mut bind_groups = Vec::<wgpu::BindGroup>::new();
        for i in 0..2 {
            bind_groups.push( device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared_buffers.sim_param_buffer.as_entire_binding(),
                    },

                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: shared_buffers.agent_buffers[i].as_entire_binding(),
                    },
                ],
                label: None,
            }));
        }

        NewDotsPipelineStage{
            bind_groups,
            pipeline
        }
        
    }
}

impl ExecutableStage for NewDotsPipelineStage
{
    fn execute(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize ) {
        
        // create a texture view from new_dots_texture
        let new_dots_texture_view = shared_buffers.new_dots_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

        command_encoder.push_debug_group("render dot for each agent");
        {
            // This is done as a render pass
            let mut rpass = command_encoder.begin_render_pass(&new_dots_render_pass_descriptor);
            rpass.set_pipeline(&self.pipeline);
            
            // render dst particles
            rpass.set_vertex_buffer(0, shared_buffers.agent_buffers[(frame_num + 1) % 2].slice(..));
            // the three instance-local vertices
            rpass.set_vertex_buffer(1, shared_buffers.new_dots_vertices_buffer.slice(..));

            rpass.set_bind_group(0, &self.bind_groups[frame_num % 2], &[]);
            
            rpass.draw(0..6, 0..shared_buffers.sim_param_data.num_agents);

        }
        command_encoder.pop_debug_group();
    }
}


struct DepositPipelineStage
{
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    width: u32,
    height: u32
}

impl DepositPipelineStage
{
    fn create(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,
        sim_param_data: &SimulationParameters,
        shared_buffers: &PipelineSharedBuffers,
        width: u32,
        height: u32
    ) -> DepositPipelineStage
    {
        // create compute pipeline
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

        
        let mut deposit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &deposit_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shared_buffers.sim_param_buffer.as_entire_binding(),
                },

                // chemo in
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.chemo_textures[0].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // new dots
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.new_dots_texture.create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // chemo out
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.chemo_textures[1].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },
            ],
            label: None,
        });

        let deposit_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("deposit"),
                bind_group_layouts: &[&deposit_bind_group_layout],
                push_constant_ranges: &[],
            });

        let deposit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("deposit.wgsl"))),
        });
            
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Deposit pipeline"),
            layout: Some(&deposit_pipeline_layout),
            module: &deposit_shader,
            entry_point: "main",
        });

        DepositPipelineStage{
            bind_group: deposit_bind_group,
            pipeline,
            width,
            height
        }
        
    }
}

impl ExecutableStage for DepositPipelineStage
{
    fn execute(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize ) 
    {
        let work_group_size = 8;

        command_encoder.push_debug_group("deposit chemo for each dot");
        {
            // compute pass
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.width/work_group_size, self.height/work_group_size, 1);
        }
        command_encoder.pop_debug_group();
    }
}

struct DiffusePipelineStage
{
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::ComputePipeline,
    width: u32,
    height: u32
}

impl DiffusePipelineStage
{
    fn create(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,
        sim_param_data: &SimulationParameters,
        shared_buffers: &PipelineSharedBuffers,
        width: u32,
        height: u32,
        sampler: &wgpu::Sampler
    ) -> DiffusePipelineStage
    {

        let diffuse_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("diffuse.wgsl"))),
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

        let mut diffuse_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &diffuse_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shared_buffers.sim_param_buffer.as_entire_binding(),
                },

                // chemo in
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.chemo_textures[1].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },

                // chemo out
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.chemo_textures[0].create_view(
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

        let diffuse_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("diffuse"),
                bind_group_layouts: &[&diffuse_bind_group_layout],
                push_constant_ranges: &[],
            });

        let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Diffuse pipeline"),
            layout: Some(&diffuse_pipeline_layout),
            module: &diffuse_shader,
            entry_point: "main",
        });

        DiffusePipelineStage{
            bind_group: diffuse_bind_group,
            pipeline: diffuse_pipeline,
            width,
            height
        }
        
    }
}

impl ExecutableStage for DiffusePipelineStage
{
    fn execute(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize ) 
    {

        let work_group_size = 8;

        command_encoder.push_debug_group("diffuse chemo");
        {
            // compute pass
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.dispatch_workgroups(self.width/work_group_size, self.height/work_group_size, 1);
        }
        command_encoder.pop_debug_group();
    }
}

struct UpdateAgentsPipelineStage
{
    bind_groups: Vec<wgpu::BindGroup>,
    pipeline: wgpu::ComputePipeline,
    width: u32,
    height: u32
}

impl UpdateAgentsPipelineStage
{
    fn create(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,
        sim_param_data: &SimulationParameters,
        shared_buffers: &PipelineSharedBuffers,
        width: u32,
        height: u32,
        sampler: &wgpu::Sampler
    ) -> UpdateAgentsPipelineStage
    {
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("compute.wgsl"))),
        });

        let update_agents_bind_group_layout =
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
                            min_binding_size: wgpu::BufferSize::new((sim_param_data.num_agents * 16) as _),
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
                            min_binding_size: wgpu::BufferSize::new((sim_param_data.num_agents * 16) as _),
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

        let mut update_bind_groups = Vec::<wgpu::BindGroup>::new();

        for i in 0..2 {
            update_bind_groups.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                layout: &update_agents_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared_buffers.sim_param_buffer.as_entire_binding(),
                    },
                    // agents in
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: shared_buffers.agent_buffers[i].as_entire_binding(),
                    },

                    // chemo in
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &shared_buffers.chemo_textures[0].create_view(
                                &wgpu::TextureViewDescriptor::default())
                        )
                    },

                    // agents out
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: shared_buffers.agent_buffers[(i + 1) % 2].as_entire_binding(), // bind to opposite buffer
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
                            &shared_buffers.control_texture.create_view(
                                &wgpu::TextureViewDescriptor::default())
                        )
                    },
                ],
                label: None,
            }));
        }

        let update_agents_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("compute"),
                bind_group_layouts: &[&update_agents_bind_group_layout],
                push_constant_ranges: &[],
            });

        let update_agents_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute pipeline"),
            layout: Some(&update_agents_pipeline_layout),
            module: &compute_shader,
            entry_point: "main",
        });

        UpdateAgentsPipelineStage{
            bind_groups: update_bind_groups,
            pipeline: update_agents_pipeline,
            width,
            height
        }
        
    }
}

impl ExecutableStage for UpdateAgentsPipelineStage
{
    fn execute(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize ) 
    {

        command_encoder.push_debug_group("update agent positions");
        {
            // compute pass
            let mut cpass =
                command_encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &self.bind_groups[frame_num % 2], &[]);
            cpass.dispatch_workgroups(shared_buffers.sim_param_data.num_agents/64, 1, 1);
        }
        command_encoder.pop_debug_group();
    }
}
struct RenderPipelineStage
{
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    width: u32,
    height: u32
}

impl RenderPipelineStage
{
    fn create(
        device: &wgpu::Device, 
        config: &wgpu::SurfaceConfiguration,
        sim_param_data: &SimulationParameters,
        shared_buffers: &PipelineSharedBuffers,
        width: u32,
        height: u32,
        sampler: &wgpu::Sampler
    ) -> RenderPipelineStage
    {
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("draw.wgsl"))),
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

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &render_bind_group_layout,
            entries: &[

                // params
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: shared_buffers.sim_param_buffer.as_entire_binding(),
                },

                // chemo
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.chemo_textures[0].create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },
                
                // control
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &shared_buffers.control_texture.create_view(
                            &wgpu::TextureViewDescriptor::default())
                    )
                },
            ],
            label: None,
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render"),
                bind_group_layouts: &[&render_bind_group_layout],
                push_constant_ranges: &[],
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

        RenderPipelineStage{
            bind_group: render_bind_group,
            pipeline: render_pipeline,
            width,
            height
        }
        
    }
}

impl RenderableStage for RenderPipelineStage
{
    fn render(&self, 
        command_encoder: &mut wgpu::CommandEncoder, 
        view: &wgpu::TextureView,
        shared_buffers: &PipelineSharedBuffers,
        frame_num: usize ) 
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
            rpass.set_pipeline(&self.pipeline);
            
            rpass.set_vertex_buffer(0, shared_buffers.agent_buffers[(frame_num + 1) % 2].slice(..));
            rpass.set_vertex_buffer(1, shared_buffers.new_dots_vertices_buffer.slice(..));

            rpass.set_bind_group(0, &self.bind_group, &[]);
            
            rpass.draw(0..6, 0..1);
        }
        
        command_encoder.pop_debug_group();
    }
}