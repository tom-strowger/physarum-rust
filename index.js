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

      let initial_values = {
        rotate_angle: 22.0,
        sense_angle: 22.0,
        sense_offset: 5.0,
        step_size: 3.0,
        decay: 0.10,
        deposit: 1.0,
      }

      wasm.add_simulation_to("physarum-div", options);

      let div = document.getElementById("physarum-div");
      div.setAttribute("style", "min-width: 900px; min-height: 600px;");

      wasm.set_foreground_colour( "4599AA" );
      wasm.set_background_colour( "EEEE89" );

      add_controls( initial_values );

    }
);

function add_controls( initial_values ) {
  
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

  ra.input.value = ra.norm( initial_values.rotate_angle );
  ra.input.dispatchEvent(new Event('input'));

  sa.input.value = sa.norm( initial_values.sense_angle );
  sa.input.dispatchEvent(new Event('input'));

  so.input.value = so.norm( initial_values.sense_offset );
  so.input.dispatchEvent(new Event('input'));

  ss.input.value = ss.norm( initial_values.step_size );
  ss.input.dispatchEvent(new Event('input'));

  dr.input.value = dr.norm( initial_values.decay );
  dr.input.dispatchEvent(new Event('input'));

  da.input.value = da.norm( initial_values.deposit );
  da.input.dispatchEvent(new Event('input'));

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

  return {
    'input': input,
    'output': output,
    'norm': normalisation,
    'denorm': denormalisation,
  }
}