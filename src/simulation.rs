
// use pipeline;
use crate::pipeline::{
    PipelineConfiguration,
    Pipeline,
};

///
/// This application is built upon a wgpu example, thus the Example struct
/// 
/// 
use winit::{
    event::{self, WindowEvent}};

use winit::{
    event_loop::{EventLoop, EventLoopBuilder, EventLoopProxy},
};
use web_sys::wasm_bindgen::JsError;
use chrono::Utc;
use web_time;  // When not targetting wasm32, this is a wrapper around std::time

use crate::framework;

// define a global frame rate limit
const FRAME_RATE_LIMIT: Option<u32> = Some(120);

pub struct Simulation {
    pipeline: Pipeline,

    running: bool,
    save: bool,
    dump: bool,

    // next time to render
    next_render_time: web_time::Instant,
}

pub fn hex_to_f32_4(hex: String) -> [f32; 4] {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap() as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap() as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap() as f32 / 255.0;
    let a = 1.0;
    [r, g, b, a]
}

use std::cell::RefCell;
#[derive(Debug)]
pub enum AppEvent {
    Pause {},
    SetSize { width: u32, height: u32 },
    SetForegroundColour{
        rgba: [f32; 4],  // RGBA
    },
    SetBackgroundColour{
        rgba: [f32; 4],  // RGBA
    },
    SetSenseAngle{ 
        sense_angle: f32,
    },
    SetSenseOffset{
        sense_offset: f32,
    },
    SetStepSize{
        step: f32,
    },
    SetRotateAngle{
        rotate_angle: f32,
    },
    SetDeposit{
        deposit: f32
    },
    SetDecay{
        decay: f32
    }
}

thread_local! {
    pub static EVENT_LOOP_PROXY: RefCell<Option<EventLoopProxy<AppEvent>>> = RefCell::new(None);
}

#[cfg(target_arch = "wasm32")]
pub fn send_event(event: AppEvent) -> Result<(), JsError> {
    EVENT_LOOP_PROXY.with_borrow(|proxy| {
        proxy
            .as_ref()
            .ok_or_else(|| {
                JsError::new(
                    "No EventLoopProxy. Did you call wasm_main() before send_custom_event()?",
                )
            })?
            .send_event(event)?;

        Ok(())
    })
}

impl framework::Example for Simulation {
    type ExampleUserEvent = AppEvent;

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
        event_loop: &EventLoop<Self::ExampleUserEvent>,
    ) -> Self {
        
        let proxy: EventLoopProxy<AppEvent> = event_loop.create_proxy();
        EVENT_LOOP_PROXY.set(Some(proxy));

        let mut pipeline = Pipeline::init(config, device, queue, 
            PipelineConfiguration::default(device, config, queue));

        let mut running = true;

        #[cfg(target_arch = "wasm32")]
        {
            let query_string = web_sys::window().unwrap().location().search().unwrap();

            if let Some(bg_colour_param) =
                framework::parse_url_query_string(&query_string, "bg_colour")
            {
                let bg_colour : String = bg_colour_param.parse().unwrap();
                pipeline.set_background_colour(hex_to_f32_4(bg_colour));
            }

            if let Some(fg_colour_param) =
                framework::parse_url_query_string(&query_string, "fg_colour")
            {
                let fg_colour : String = fg_colour_param.parse().unwrap();
                pipeline.set_foreground_colour(hex_to_f32_4(fg_colour));
            }

            if let Some(rotate_angle_param) = 
                framework::parse_url_query_string(&query_string, "rotate_angle")
            {
                let rotate_angle : f32 = rotate_angle_param.parse().unwrap();
                pipeline.set_rotate_angle(rotate_angle);
            }

            if let Some(sense_angle_param) = 
                framework::parse_url_query_string(&query_string, "sense_angle")
            {
                let rotate_angle : f32 = sense_angle_param.parse().unwrap();
                pipeline.set_sense_angle(rotate_angle);
            }

            if let Some(sense_offset_param) = 
                framework::parse_url_query_string(&query_string, "sense_offset")
            {
                let rotate_angle : f32 = sense_offset_param.parse().unwrap();
                pipeline.set_sense_offset(rotate_angle);
            }

            if let Some(step_size_param) = 
                framework::parse_url_query_string(&query_string, "step_size")
            {
                let rotate_angle : f32 = step_size_param.parse().unwrap();
                pipeline.set_step_size(rotate_angle);
            }

            if let Some(num_agents_param) = 
                framework::parse_url_query_string(&query_string, "num_agents")
            {
                let rotate_angle : u32 = num_agents_param.parse().unwrap();
                pipeline.set_num_agents(rotate_angle);
            }

            if let Some(decay_param) = 
                framework::parse_url_query_string(&query_string, "decay")
            {
                let rotate_angle : f32 = decay_param.parse().unwrap();
                pipeline.set_decay(rotate_angle);
            }

            if let Some(deposit_param) = 
                framework::parse_url_query_string(&query_string, "deposit")
            {
                let rotate_angle : f32 = deposit_param.parse().unwrap();
                pipeline.set_deposit(rotate_angle);
            }

            if let Some(running_param) = 
                framework::parse_url_query_string(&query_string, "running")
            {
                let running_value : bool = running_param.parse().unwrap();
                running = running_value;
            }
        }

        // returns Example struct and No encoder commands
        Simulation {
            pipeline: pipeline,
            running: running,
            save: false,
            dump: false,

            next_render_time: web_time::Instant::now(),
        }
    }

    fn handle_user_event(&mut self, event: Self::ExampleUserEvent) {
        log::info!( "received user event {:?}", event );
        match event {
            AppEvent::SetSize { width, height } 
                => {
            },
            AppEvent::Pause {} => {
            },
            AppEvent::SetForegroundColour { rgba } => {
                self.pipeline.set_foreground_colour( rgba );
            },
            AppEvent::SetBackgroundColour { rgba } => {
                self.pipeline.set_background_colour( rgba );
            },
            AppEvent::SetSenseAngle { sense_angle } => {
                self.pipeline.set_sense_angle( sense_angle );
            },
            AppEvent::SetSenseOffset { sense_offset } => {
                self.pipeline.set_sense_offset( sense_offset );
            },
            AppEvent::SetStepSize { step } => {
                self.pipeline.set_step_size( step );
            },
            AppEvent::SetRotateAngle { rotate_angle } => {
                self.pipeline.set_rotate_angle( rotate_angle );
            },
            AppEvent::SetDeposit { deposit } => {
                self.pipeline.set_deposit( deposit );
            },
            AppEvent::SetDecay { decay } => {
                self.pipeline.set_decay( decay );
            },
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
        
        let time_to_render = FRAME_RATE_LIMIT.is_none() || web_time::Instant::now() >= self.next_render_time;

        if time_to_render
        {
            if let Some(frame_rate_limit) = FRAME_RATE_LIMIT {
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

            self.pipeline.save_chemo_image(
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hex_to_f32_4() {
        assert_eq!(hex_to_f32_4("#00000000".to_string()), [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(hex_to_f32_4("#FF0000FF".to_string()), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(hex_to_f32_4("#00FF00FF".to_string()), [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(hex_to_f32_4("#0000FFFF".to_string()), [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(hex_to_f32_4("#FFFFFFFF".to_string()), [1.0, 1.0, 1.0, 1.0]);
    }
}
        