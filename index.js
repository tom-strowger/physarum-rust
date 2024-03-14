// Load the package and run the simulation
let mod;
let controls;

import('./pkg')
    .then(wasm => {
      mod = wasm;

      let options = {
        width: 900,
        height: 600
      }

      wasm.add_simulation_to("physarum-canvas", options);

      // Todo: init the loop synchronously when the module imports so this doesn't have to be done with a timer
      document.timer = setInterval( function() { wasm.set_foreground_colour( "4599AA" ); }, 1000 );
      document.timer = setInterval( function() { wasm.set_background_colour( "EEEE89" ); }, 1000 );

      add_controls();

    }
);

function add_controls() {
  
  let container = document.getElementById('simulation-controls');
  if (!container) throw new Error('Unable to find #container in dom');

  let table = document.createElement('table');

  let angle_denorm =  function(x){ return Math.pow( x, 3.0 ) * 360.0; };
  let angle_norm = function(x){ return Math.pow( x / 360.0, 1.0 / 3.0 )  };

  let ra = add_control_to_table("Rotate angle", table, 
    angle_denorm, 
    angle_norm,
    function(x){ mod.set_rotate_angle(x) } );
  let sa = add_control_to_table("Sense angle", table, 
    angle_denorm, 
    angle_norm,
    function(x){ mod.set_sense_angle(x) } );
  let so = add_control_to_table("Sense offset", table, 
    function(x){ return x * 10; }, 
    function(x){ return x / 10; },
    function(x){ mod.set_sense_offset(x) } );
  let ss = add_control_to_table("Step size", table,
    function(x){ return x * 10; }, 
    function(x){ return x / 10; },
    function(x){ mod.set_step_size(x) } );
  let dr = add_control_to_table("Decay ratio", table, 
    function(x){ return  Math.pow( x, 2.0 );},
    function(x){ return Math.pow( x, 0.5 ); },
    function(x){ mod.set_decay(x) } );
  let da = add_control_to_table("Deposit amount", table, 
    function(x){ return x * 5;}, 
    function(x){ return x / 5; },
    function(x){ mod.set_deposit(x) } );

  ra.value = "0.30";
  sa.value = "0.35"
  so.value = "0.2";
  ss.value = "0.2";
  dr.value = "0.25";
  da.value = "0.12";

  container.appendChild(table);

}

function add_control_to_table( name, table, denormalisation, normalisation,
  on_change = function(x){} ) {
  
  let row = document.createElement('tr');
  let label = document.createElement('td');
  label.innerHTML = name;

  let input = document.createElement('input');
  input.type = "range";
  input.min = "0";
  input.max = "1.0";
  input.step = "0.001";
  input.class="slider";

  let output = document.createElement('td');
  output.innerHTML = input.value;

  input.oninput = function() {
    let denorm_value = denormalisation( parseFloat(this.value) ).toFixed(1)
    on_change( denorm_value );
    output.innerHTML = denorm_value;
  }

  row.appendChild(label);
  row.appendChild(input);
  row.appendChild(output);
  table.appendChild(row);

  return input;
}