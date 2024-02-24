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
use web_time;  // When not targetting wasm32, this is a wrapper around std::time

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
    next_render_time: web_time::Instant,
}

fn hex_to_f32_4(hex: String) -> [f32; 4] {
    let hex = hex.trim_start_matches('#');
    let r = u8::from_str_radix(&hex[0..2], 16).unwrap() as f32 / 255.0;
    let g = u8::from_str_radix(&hex[2..4], 16).unwrap() as f32 / 255.0;
    let b = u8::from_str_radix(&hex[4..6], 16).unwrap() as f32 / 255.0;
    let a = 1.0;
    [r, g, b, a]
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
        Application {
            pipeline: pipeline,
            running: running,
            save: false,
            dump: false,

            next_render_time: web_time::Instant::now(),
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


struct OnlineTestRunner {
    pipeline: Pipeline
}

impl framework::Example for OnlineTestRunner {
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
        OnlineTestRunner {
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

    let mut online_test_mode = false;
    
    #[cfg(not(target_arch = "wasm32"))]
    for a in &args[1..]{
        match a.as_str()
        {
            "--online_test" => online_test_mode = true,
            _ => println!("Unhandled argument {}",a),
        }
    }

    let mut logical_width = LOGICAL_WIDTH;
    let mut logical_height = LOGICAL_HEIGHT;
    #[cfg(target_arch = "wasm32")]
    {
        let query_string = web_sys::window().unwrap().location().search().unwrap();

        if let Some(width_param) =
            framework::parse_url_query_string(&query_string, "width")
        {
            let width : u32 = width_param.parse().unwrap();
            logical_width = width;
        }

        if let Some(height_param) =
            framework::parse_url_query_string(&query_string, "height")
        {
            let height : u32 = height_param.parse().unwrap();
            logical_height = height;
        }
        
    }

    if online_test_mode {
        // @todo Implement a custom test harness for these tests (using just the main thread)
        framework::run::<OnlineTestRunner>("Physarum", (logical_width, logical_height));
    } else {
        framework::run::<Application>("Physarum", (logical_width, logical_height));
    }

}

/// These are a group of tests that can run "online" (with a real pipeline/wgpu adapter)
mod online_tests {
    use std::iter::zip;

    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;
    use serde::de;
    use wgpu::{util::DeviceExt};

    #[derive(Copy, Clone, Debug)]
    struct Point
    {
        x: u32,
        y: u32,
    }

    #[derive(Copy, Clone, Debug)]
    struct Size
    {
        width: u32,
        height: u32,
    }

    struct Image
    {
        size: Size,
        data: Vec<u8>
    }

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


    fn create_image(
        image_size: Size,
        colour: [u8; 4],) -> Image
    {
        let mut img_data = Vec::new();
        img_data.reserve((image_size.width * image_size.height * 4) as usize);

        for _ in 0..(image_size.width * image_size.height){
            for i in 0..colour.len(){
                img_data.push( colour[i] );
            }
        }
        Image {
            size: image_size,
            data: img_data
        }
    }

    fn add_horitontal_gradient_to_image(
        image: &mut Image,
        gradient_size: Size,
        gradient_pos: Point,
        left_colour: [u8; 4],
        right_colour: [u8; 4], )
    {
        for x in 0..gradient_size.width {

            let mix = x as f32 / gradient_size.width as f32;
            for y in 0..gradient_size.height {

                let mix = x as f32 / gradient_size.width as f32;

                let image_x = x + gradient_pos.x;
                let image_y = y + gradient_pos.y;

                if image_x > 0 && image_x < image.size.width &&
                    image_y > 0 && image_y < image.size.height {
                    
                    let image_index = ( image_x + image_y * image.size.width ) as usize;
                    
                    // Copy over the rgba data, performing a linear mix
                    for (i,(l,r)) in zip(left_colour, right_colour).enumerate() {
                        let pixel_rgb_channel_index = image_index * 4 + i;
                        image.data[pixel_rgb_channel_index] = ( l as f32 * (1.0-mix) + r as f32 * mix ) as u8;
                    }
        
                }
            }
        }
    }

    fn create_image_with_horizontal_gradient(
        image_size: Size,
        gradient_size: Size,
        gradient_pos: Point,
        left_colour: [u8; 4],
        right_colour: [u8; 4], ) -> Image
    {
        let mut image = create_image(image_size, [0,0,0,0]);
        add_horitontal_gradient_to_image( &mut image, gradient_size, gradient_pos, left_colour, right_colour);
        image
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

    fn test_agents_deposit(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_all_agents_turn_to_face_ascending_gradient----");

        // GIVEN
        // A chemo texture which is dark to light from left to right
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        let chemo_image = create_image_with_horizontal_gradient(
            Size{ width, height }, 
            Size{ width, height },
            Point { x: 0, y: 0 },
            [0, 0, 0, 0], 
            [255, 255, 255, 255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_image.data);

        let half_point_width = 0.5 / ( width as f32 );
        let half_point_height = 0.5 / ( height as f32 );

        // And four agents at different positions
        let mut agents = vec![
            AgentData::init( 0.1, 0.8, 0.52 ),
            AgentData::init( 0.2, 0.1, 0.98 ),
            AgentData::init( 0.6, 0.8, 0.02 ),
            AgentData::init( 0.7, 0.1, 0.23 )];

        // Put each agent squarely in the middle of a fragment
        for a in &mut agents {
            a.pos_x -= half_point_width;
            a.pos_y -= half_point_height;
        }
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            agents.to_vec());

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // The agent positions update correctly, they should all turn and point to increasing chemo.  Which is left to right
        let new_dots_data = pipeline.get_shared_buffers().get_new_dots_data(device, queue);

        let new_dot_positions : Vec<usize> = new_dots_data.chunks(4).enumerate()
            .map(|(i,x)| if x[0]>128{ Some(i) }else{ None })
            .filter(|x|x.is_some())
            .map(|x|x.unwrap())
            .collect();

        let mut agent_positions : Vec<usize> = agents.iter().map(|a|{
            let x = (a.pos_x * width as f32) as u32;
            let y = (a.pos_y * height as f32) as u32;
            (x + y * width) as usize
        }).collect();

        agent_positions.sort();

        assert!( new_dot_positions == agent_positions );

        // AFTER
        // Reset the frame number for the next test
        pipeline.reset_frame_num();

        println!("    \\-- OK ");

    }

    fn test_all_agents_turn_to_face_ascending_gradient(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_all_agents_turn_to_face_ascending_gradient----");

        // GIVEN
        // A chemo texture which is dark to light from left to right
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        let chemo_image = create_image_with_horizontal_gradient(
            Size{ width, height }, 
            Size{ width, height },
            Point { x: 0, y: 0 },
            [0, 0, 0, 0], 
            [255, 255, 255, 255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_image.data);

        // And four agents facing in different directions
        // The agents sense ahead and slightly to the left and right, 
        // so they are postioned such that there is a large gradient
        // between left and right. That way they will turn.
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            vec![
            AgentData::init( 0.2, 0.8, 0.52 ),   // facing descending gradient
            AgentData::init( 0.2, 0.1, 0.98 ),   // facing descending gradient
            AgentData::init( 0.7, 0.8, 0.02 ),   // facing ascending gradient
            AgentData::init( 0.7, 0.1, 0.23 )]); // facing ascending gradient

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // The agent positions update correctly, they should all turn and point to increasing chemo.  Which is left to right
        let agent_data = pipeline.get_shared_buffers().get_agent_data(device, queue, 1);
        assert!(agent_data[0].heading >= 0.00 || agent_data[0].heading <= 0.50 );
        assert!(agent_data[1].heading >= 0.00 || agent_data[1].heading <= 0.50 );
        assert!(agent_data[2].heading >= 0.00 || agent_data[1].heading <= 0.50 );
        assert!(agent_data[3].heading >= 0.00 || agent_data[1].heading <= 0.50 );

        // AFTER
        // Reset the frame number for the next test
        pipeline.reset_frame_num();

        println!("    \\-- OK ");

    }

    fn test_only_the_agent_that_should_turn_does_top_left(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_only_the_agent_that_should_turn_does_top_left----");

        // GIVEN
        // A chemo texture which is dark to light from left to right, but only covers the top-left quarter of the screen
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        // let chemo_image = create_image( Size{ width, height }, [0, 0, 0, 0] );
        let chemo_image = create_image_with_horizontal_gradient(
            Size{ width, height }, 
            Size{ width: width / 2, height: height / 2 },
            Point { x: 0, y: 0 },
            [0, 0, 0, 0], 
            [255, 255, 255, 255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_image.data);

        // And four agents facing to the left
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            vec![
            AgentData::init( 0.2, 0.1, 0.52 ),   // top left
            AgentData::init( 0.2, 0.8, 0.52 ),   // bottom left
            AgentData::init( 0.7, 0.1, 0.52 ),   // top right
            AgentData::init( 0.7, 0.8, 0.52 )]); // bottom right

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // Only the top-left agent should change as it detects an increase gradient in a different direction
        let agent_data = pipeline.get_shared_buffers().get_agent_data(device, queue, 1);
        assert!(agent_data[0].heading < 0.52 );
        assert!(agent_data[1].heading == 0.52 );
        assert!(agent_data[2].heading == 0.52 );
        assert!(agent_data[3].heading == 0.52 );

        // AFTER
        // Reset the frame number for the next test
        pipeline.reset_frame_num();

        println!("    \\-- OK ");

    }

    fn test_only_the_agent_that_should_turn_does_bottom_left(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_only_the_agent_that_should_turn_does_bottom_left----");

        // GIVEN
        // A chemo texture which is dark to light from left to right, but only covers the top-left quarter of the screen
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        let chemo_image = create_image_with_horizontal_gradient(
            Size{ width, height }, 
            Size{ width: width / 2, height: height / 2 },
            Point { x: 0, y: height / 2 },
            [0, 0, 0, 0], 
            [255, 255, 255, 255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_image.data);

        // And four agents facing to the left
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            vec![
            AgentData::init( 0.2, 0.1, 0.52 ),   // top left
            AgentData::init( 0.2, 0.8, 0.52 ),   // bottom left
            AgentData::init( 0.7, 0.1, 0.52 ),   // top right
            AgentData::init( 0.7, 0.8, 0.52 )]); // bottom right

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // Only the top-left agent should change as it detects an increase gradient in a different direction
        let agent_data = pipeline.get_shared_buffers().get_agent_data(device, queue, 1);
        assert!(agent_data[0].heading == 0.52 );
        assert!(agent_data[1].heading < 0.52 );
        assert!(agent_data[2].heading == 0.52 );
        assert!(agent_data[3].heading == 0.52 );

        // AFTER
        // Reset the frame number for the next test
        pipeline.reset_frame_num();

        println!("    \\-- OK ");

    }

    fn test_only_the_agent_that_should_turn_does_top_right(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_only_the_agent_that_should_turn_does_top_right----");

        // GIVEN
        // A chemo texture which is dark to light from left to right, but only covers the top-left quarter of the screen
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        // let chemo_image = create_image( Size{ width, height }, [0, 0, 0, 0] );
        let chemo_image = create_image_with_horizontal_gradient(
            Size{ width, height }, 
            Size{ width: width / 2, height: height / 2 },
            Point { x: width / 2, y: 0 },
            [0, 0, 0, 0], 
            [255, 255, 255, 255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_image.data);

        // And four agents facing to the left
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            vec![
            AgentData::init( 0.2, 0.1, 0.52 ),   // top left
            AgentData::init( 0.2, 0.8, 0.52 ),   // bottom left
            AgentData::init( 0.7, 0.1, 0.52 ),   // top right
            AgentData::init( 0.7, 0.8, 0.52 )]); // bottom right

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // Only the top-left agent should change as it detects an increase gradient in a different direction
        let agent_data = pipeline.get_shared_buffers().get_agent_data(device, queue, 1);
        assert!(agent_data[0].heading == 0.52 );
        assert!(agent_data[1].heading == 0.52 );
        assert!(agent_data[2].heading < 0.52 );
        assert!(agent_data[3].heading == 0.52 );

        // AFTER
        // Reset the frame number for the next test
        pipeline.reset_frame_num();

        println!("    \\-- OK ");

    }

    fn test_only_the_agent_that_should_turn_does_bottom_right(
        pipeline: &mut Pipeline,
        device : &wgpu::Device,
        queue: &wgpu::Queue)
    {
        println!("----test_only_the_agent_that_should_turn_does_bottom_right----");

        // GIVEN
        // A chemo texture which is dark to light from left to right, but only covers the top-left quarter of the screen
        let (width, height) = pipeline.get_shared_buffers().get_chemo_texture_width_height();
        let chemo_image = create_image_with_horizontal_gradient(
            Size{ width, height }, 
            Size{ width: width / 2, height: height / 2 },
            Point { x: width / 2, y: height / 2 },
            [0, 0, 0, 0], 
            [255, 255, 255, 255]);
        pipeline.get_shared_buffers().upload_to_chemo_texture(device, queue, width, height, &chemo_image.data);

        // And four agents facing to the left
        pipeline.get_shared_buffers().set_agent_data(
            device, queue, 0,
            vec![
            AgentData::init( 0.2, 0.1, 0.52 ),   // top left
            AgentData::init( 0.2, 0.8, 0.52 ),   // bottom left
            AgentData::init( 0.7, 0.1, 0.52 ),   // top right
            AgentData::init( 0.7, 0.8, 0.52 )]); // bottom right

        // WHEN
        // I compute a step
        pipeline.compute_step(device, queue);

        // THEN
        // Only the top-left agent should change as it detects an increase gradient in a different direction
        let agent_data = pipeline.get_shared_buffers().get_agent_data(device, queue, 1);
        assert!(agent_data[0].heading == 0.52 );
        assert!(agent_data[1].heading == 0.52 );
        assert!(agent_data[2].heading == 0.52 );
        assert!(agent_data[3].heading < 0.52 );

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

        test_agents_deposit( pipeline, device, queue );

        test_all_agents_turn_to_face_ascending_gradient( pipeline, device, queue );
        test_only_the_agent_that_should_turn_does_top_left( pipeline, device, queue );
        test_only_the_agent_that_should_turn_does_bottom_left(pipeline, device, queue );
        test_only_the_agent_that_should_turn_does_top_right( pipeline, device, queue );
        test_only_the_agent_that_should_turn_does_bottom_right(pipeline, device, queue );

        println!("Testing done");
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
        