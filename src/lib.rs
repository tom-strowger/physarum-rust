
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
    framework::run::<Simulation>("Physarum", (1280, 800), container_name );
}

pub fn test() {
    println!("Test");
}