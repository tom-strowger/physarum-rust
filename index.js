// Load the package and run the simulation
import('./pkg')
    .then(wasm => {
      console.log(wasm);
      wasm.add_simulation_to("physarum-canvas");
    }
    );