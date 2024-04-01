use std::env;


///
/// This application is built upon a wgpu example, thus the Example struct
/// 
/// 
mod pipeline;

mod framework;

mod simulation;

use simulation::Simulation;

#[cfg(not(target_arch = "wasm32"))]
mod test_runner;

use env_logger;

/// run example
fn main() {
    env_logger::init();

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

    // A3 = 3508 x 4960
    // A4 = 2480 x 3508

    let mut logical_width = 1280;
    let mut logical_height = 800;

    if online_test_mode {
        // @todo Implement a custom test harness for these tests (using just the main thread)
#[cfg(not(target_arch = "wasm32"))]
        framework::run::<test_runner::OnlineTestRunner>("Physarum", (logical_width, logical_height));
    } else {
        framework::run::<Simulation>("Physarum", (logical_width, logical_height));
    }

}
