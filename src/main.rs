use std::env;

use pipeline::{
    AgentData,
    PipelineConfiguration};

///
/// This application is built upon a wgpu example, thus the Example struct
/// 
/// 
use winit::event::{self, WindowEvent};
use chrono::Utc;

mod pipeline;
use crate::pipeline::Pipeline;

#[path = "./framework.rs"]
mod framework;

// define a global frame rate limit
const FRAME_RATE_LIMIT: Option<u32> = Some(120);

struct Application {
    pipeline: Pipeline,

    running: bool,
    save: bool,
    dump: bool,

    // next time to render
    next_render_time: std::time::Instant,
}

const LOGICAL_WIDTH : u32 = 1280;
const LOGICAL_HEIGHT : u32 = 800;

impl framework::Example for Application {
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

        // returns Example struct and No encoder commands
        Application {
            pipeline: Pipeline::init(config, device, queue, 
                PipelineConfiguration::default(device, config, queue)),

            running: true,
            save: false,
            dump: false,

            next_render_time: std::time::Instant::now(),
        }
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
                let alpha = self.pipeline.get_control_alpha();
                if alpha == 0.0 {
                    self.pipeline.set_control_alpha( 0.2 );
                }
                else if alpha == 0.2 {
                    self.pipeline.set_control_alpha( 1.0 );
                }
                else {
                    self.pipeline.set_control_alpha( 0.0 );
                }
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

        self.pipeline.resize(w, h);
    }

    /// render is called each frame, dispatching compute groups proportional
    ///   a TriangleList draw call for all NUM_PARTICLES at 3 vertices each
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _: &framework::Spawner,
        _: &mut bool
    ) {
        
        let time_to_render = FRAME_RATE_LIMIT.is_none() || std::time::Instant::now() >= self.next_render_time;

        if time_to_render
        {
            if let(Some(frame_rate_limit)) = FRAME_RATE_LIMIT {
                self.next_render_time += std::time::Duration::from_millis((1000.0 / frame_rate_limit as f32) as u64);
            }

            if self.running
            {
                self.pipeline.compute_step( device, queue);
                    
            }
        }

        // render to view
        self.pipeline.render(view, device, queue);

        // save the chemo texture to an image
        if self.save
        {
            self.save = false;

            let now = Utc::now();

            self.pipeline.save_image(
                device, 
                queue, 
                format!( "chemo_{}",  now.format("%Y%m%d_%H%M%S") ).as_str() );
        }

        if self.dump
        {
            self.dump = false;

            self.pipeline.dump_debug_textures(device, queue);
        }

    }
}


struct TestRunner {
    pipeline: Pipeline
}

impl framework::Example for TestRunner {
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

        // returns Example struct and No encoder commands
        TestRunner {
            pipeline: Pipeline::init(config, device, queue, 
                PipelineConfiguration::default(device, config, queue))
        }
    }

    /// update is called for any WindowEvent not handled by the framework
    fn update(&mut self, event: winit::event::WindowEvent) {
        //empty
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

        self.pipeline.resize(w, h);
    }

    /// render is called each frame, dispatching compute groups proportional
    ///   a TriangleList draw call for all NUM_PARTICLES at 3 vertices each
    fn render(
        &mut self,
        view: &wgpu::TextureView,
        texture: &wgpu::Texture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _: &framework::Spawner,
        should_exit: &mut bool
    ) {
        online_tests::test_pipeline( &mut self.pipeline, device, queue  );
        *should_exit = true;
    }
}

/// run example
fn main() {
    // parse command line arguments
    let args: Vec<String> = env::args().collect();

    let mut test_mode = false;

    for a in &args[1..]{
        match a.as_str()
        {
            "--test" => test_mode = true,
            _ => println!("Unhandled argument {}",a),
        }
    }

    if test_mode {
        framework::run::<TestRunner>("Physarum", (LOGICAL_WIDTH, LOGICAL_HEIGHT));
    } else {
        framework::run::<Application>("Physarum", (LOGICAL_WIDTH, LOGICAL_HEIGHT));
    }
}

/// These are a group of tests that can run "online" (with a real pipeline/wgpu adapter)
mod online_tests {
    use std::iter::zip;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use wgpu::{util::DeviceExt};

    fn create_image_with_vertical_line(
        width: u32,
        height: u32) -> Vec<u8>
    {
        let black_pixel : Vec<u8> = [ 0, 0, 0, 0].to_vec();
        let white_pixel : Vec<u8> = [ 255, 255, 255, 255].to_vec();

        // scale the texture data to fit the texture
        let mut img_data = Vec::new();
        img_data.reserve((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {

                if y == width / 2 {
                    for &d in &white_pixel{
                        img_data.push(d);
                    }
                }
                else {
                    for &d in &black_pixel{
                        img_data.push(d);
                    }
                }

            }
        }

        (img_data)
    }

    fn create_image_with_horizontal_gradient(
        width: u32,
        height: u32,
        left_colour: [u8; 4],
        right_colour: [u8; 4] ) -> Vec<u8>
    {
        // scale the texture data to fit the texture
        let mut img_data = Vec::new();
        img_data.reserve((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {

                let mix = x as f32 / width as f32;
                for (l,r) in zip(left_colour, right_colour) {
                    img_data.push( ( l as f32 * (1.0-mix) + r as f32 * mix ) as u8 );
                }

            }
        }

        (img_data)
    }

    fn create_texture_with_vertical_line(
        device : &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32) -> wgpu::Texture
    {
        let image_data = online_tests::create_image_with_vertical_line( width, height );

        let texture_descriptor = wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: width,
                height: height,
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
        let texture = device.create_texture_with_data(queue, &texture_descriptor, &image_data );

        (texture)
    }

    fn test_pipeline_agent_step(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_pipeline_agent_step----");

        // GIVEN
        // A chemo texture which is dark to light from left to right
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        let chemo_data = create_image_with_horizontal_gradient(width, height, [0,0,0,0], [255,255,255,255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_data);

        // And four agents facing in different directions
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            vec![
            AgentData::init( 0.2, 0.8, 0.52 ),   // facing descending gradient
            AgentData::init( 0.2, 0.1, 0.98 ),   // facing descending gradient
            AgentData::init( 0.4, 0.8, 0.02 ),   // facing ascending gradient
            AgentData::init( 0.4, 0.1, 0.23 )]); // facing ascending gradient

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // The agent positions update correctly, they should all turn and point to increasing chemo.  Which is left to right
        let agent_data = pipeline.get_shared_buffers().get_agent_data(device, queue, 1);
        // println!("Agents = \n- {:?} \n- {:?} \n- {:?} \n- {:?}", agent_data[0], agent_data[1], agent_data[2], agent_data[3] );
        assert!(agent_data[0].heading >= 0.00 || agent_data[0].heading <= 0.50 );
        assert!(agent_data[1].heading >= 0.00 || agent_data[1].heading <= 0.50 );
        assert!(agent_data[2].heading >= 0.00 || agent_data[1].heading <= 0.50 );
        assert!(agent_data[3].heading >= 0.00 || agent_data[1].heading <= 0.50 );

        // AFTER
        // Reset the frame number for the next test
        pipeline.reset_frame_num();

        println!("    \\-- OK ");

    }

    pub fn test_pipeline(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        println!("Beginning tests");

        test_pipeline_agent_step( pipeline, device, queue );

        println!("Testing done");
    }
}