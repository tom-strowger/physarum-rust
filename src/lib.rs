
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
    let logical_width = options.width.unwrap_or(1280);
    let logical_height = options.height.unwrap_or(800);
    framework::run::<Simulation>("Physarum", (logical_width,logical_height), container_name );
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn resize(width: u32, height:u32) -> Result<(), JsError> {
    simulation::send_event( simulation::AppEvent::SetSize { width, height } )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_sense_angle(x: f32) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetSenseAngle { sense_angle: x }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_sense_offset(x: f32) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetSenseOffset { sense_offset: x }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_step_size(x: f32) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetStepSize { step: x }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_rotate_angle(x: f32) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetRotateAngle { rotate_angle: x }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_deposit(x: f32) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetDeposit { deposit: x }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_decay(x: f32) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetDecay { decay: x }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_foreground_colour(x: String) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetForegroundColour { rgba: simulation::hex_to_f32_4(x) }
    )
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub fn set_background_colour(x: String) -> Result<(), JsError> {
    simulation::send_event( 
        simulation::AppEvent::SetBackgroundColour { rgba: simulation::hex_to_f32_4(x) }
    )
}
