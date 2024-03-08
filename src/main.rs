use std::env;


///
/// This application is built upon a wgpu example, thus the Example struct
/// 
/// 
use winit::event::{self, WindowEvent};
use chrono::Utc;
use web_time;  // When not targetting wasm32, this is a wrapper around std::time

mod pipeline;
use crate::pipeline::Pipeline;

mod framework;

mod simulation;

mod test_runner;

use pipeline::{
    AgentData,
    PipelineConfiguration};

use simulation::Simulation;
use test_runner::OnlineTestRunner;
use framework::Example;


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

    let mut logical_width = 1280;
    let mut logical_height = 800;
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
#[cfg(not(target_arch = "wasm32"))]
        framework::run::<OnlineTestRunner>("Physarum", (logical_width, logical_height));
    } else {
        framework::run::<Simulation>("Physarum", (logical_width, logical_height));
    }

}
