// Load the package and run the simulation
import('./pkg')
    .then(wasm => {
      console.log(wasm);

      let options = {
        fg_colour: "EEFF89",
        bg_colour: "4599AA",
        width: 900,
        height: 600
      }

      wasm.add_simulation_to("physarum-canvas", options);

      window.setTimeout(function(){ wasm.resize(800, 800); }, 100);


      
    }
    );