
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod simulation;

mod pipeline;
use pipeline::{
    AgentData,
    PipelineConfiguration};

use simulation::Simulation;

mod framework;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn add_simulation_to( container_name : String ){

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
    
    framework::run::<Simulation>("Physarum", (logical_width,logical_height), container_name );
}