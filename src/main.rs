use pipeline::PipelineConfiguration;
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

struct Example {
    pipeline: Pipeline,

    running: bool,
    save: bool,
    dump: bool,

    // next time to render
    next_render_time: std::time::Instant,
}

const LOGICAL_WIDTH : u32 = 1280;
const LOGICAL_HEIGHT : u32 = 800;

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

        // returns Example struct and No encoder commands
        Example {
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

/// run example
fn main() {
    framework::run::<Example>("Physarum", (LOGICAL_WIDTH, LOGICAL_HEIGHT));
}
