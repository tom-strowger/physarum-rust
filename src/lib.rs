
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

mod simulation;
mod pipeline;

use simulation::Simulation;

use serde::{Serialize, Deserialize};

mod framework;


#[derive(Serialize, Deserialize)]
pub struct Options {
    #[serde(default)]
    pub width : Option<u32>,
    #[serde(default)]
    pub height : Option<u32>,
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn add_simulation_to( container_name : String, val: JsValue ){

    framework::init_web_log();

    let options: Options = serde_wasm_bindgen::from_value(val).unwrap();
    let mut logical_width = options.width.unwrap_or(1280);
    let mut logical_height = options.height.unwrap_or(800);
    framework::run::<Simulation>("Physarum", (logical_width,logical_height), container_name );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn resize(width: u32, height:u32) -> Result<(), JsError> {
    framework::send_event( framework::AppEvent::SetSize { width, height } )
}