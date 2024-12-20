// Load the package and run the simulation
let mod;
let controls;

import('./pkg')
    .then(wasm => {
      mod = wasm;

      // Optionally get the width and height from the page parameters
      let url = new URL(window.location.href);
      let width = parseInt(url.searchParams.get("width")||"600");
      let height = parseInt(url.searchParams.get("height")||"600");

      let options = {
        width: width,
        height: height,
      }

      let initial_values = {
        rotate_angle: 22.0,
        sense_angle: 22.0,
        sense_offset: 5.0,
        step_size: 3.0,
        decay: 0.10,
        deposit: 1.0,
        number_of_agents: 1 << 20,
      }

      wasm.add_simulation_to("physarum-div", options);

      let div = document.getElementById("physarum-div");
      div.setAttribute("style", "min-width: " + options.width + "px; min-height: "  + options.height + "px;");

      wasm.set_foreground_colour( "74A19A" );
      wasm.set_background_colour( "EBD999" );

      add_controls( initial_values );

    }
);

function add_controls( initial_values ) {
  
  let container = document.getElementById('simulation-controls');
  if (!container) throw new Error('Unable to find #container in dom');

  let table = document.createElement('table');
  let title_row = document.createElement('tr');
  let title_data = document.createElement('td');
  let title = document.createElement('h2');
  title.textContent = "Parameters";
  title_data.appendChild(title);
  title_row.appendChild(title_data);
  table.appendChild(title_row);

  let angle_denorm =  function(x){ return (Math.pow( x, 3.0 ) * 360.0).toFixed(1); };
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
  function(x){ return (x * 10).toFixed(1); }, 
  function(x){ return (x / 10); },
    function(x){ mod.set_sense_offset(x) } );
  let ss = add_control_to_table("Step size", table,
    function(x){ return (x * 10).toFixed(1); }, 
    function(x){ return (x / 10); },
    function(x){ mod.set_step_size(x) } );
  let dr = add_control_to_table("Decay ratio", table, 
    function(x){ return  Math.pow( x, 2.0 ).toFixed(3); },
    function(x){ return Math.pow( x, 0.5 ); },
    function(x){ mod.set_decay(x) } );
  let da = add_control_to_table("Deposit amount", table, 
    function(x){ return (x * 5).toFixed(2);}, 
    function(x){ return x / 5 },
    function(x){ mod.set_deposit(x) } );

  let na = add_control_to_table("Number of particles", table, 
    function(x){
      return 1 << x;
    }, 
    function(x){
      return Math.log2(x); 
    },
    function(x){ mod.set_number_of_agents(x) },
    10, 21, 1 );

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

  na.input.value = na.norm( initial_values.number_of_agents );
  na.input.dispatchEvent(new Event('input'));


  let url = new URL(window.location.href);
  let extra_controls = (url.searchParams.get("extra_controls")||"0") == "1";

  if( extra_controls )
  {
    let bc = add_colour_control_to_table("Background colour", table, 
      function(x){ mod.set_background_colour(x) } );
    bc.input.value = "#EBD999";
  
    let fc = add_colour_control_to_table("Foreground colour", table, 
    function(x){ mod.set_foreground_colour(x) } );
    fc.input.value = "#74A19A";
  }

  let controls_title_row = document.createElement('tr');
  let control_title = document.createElement('h2');
  control_title.textContent = "Controls";
  controls_title_row.appendChild(control_title);
  table.appendChild(controls_title_row);

  let pause_button_row = document.createElement('tr');
  let pause_button_td = document.createElement('td');
  let pause_button = document.createElement('button');
  pause_button.textContent = "Pause";
  pause_button.onclick = function() {
    pause = pause_button.textContent == "Pause";
    mod.set_pause( pause );
    pause_button.textContent = pause ? "Resume" : "Pause";
  }
  pause_button.setAttribute("style", "width: 100px;");
  pause_button_td.appendChild(pause_button);
  pause_button_row.appendChild(pause_button_td);
  table.appendChild(pause_button_row);

  let save_button_row = document.createElement('tr');
  let save_button_td = document.createElement('td');
  let save_button = document.createElement('button');
  save_button.textContent = "Save";
  save_button.onclick = function() {
    var canvas = document.getElementById('physarum-div');
    var dataURL = canvas.toDataURL("image/png", 1.0);
    downloadImage(dataURL, 'physarum.png');
  }

  function downloadImage(data, filename = 'untitled.jpeg') {
      var a = document.createElement('a');
      a.href = data;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
  }
  save_button.setAttribute("style", "width: 100px;");
  save_button_td.appendChild(save_button);
  save_button_row.appendChild(save_button_td);
  table.appendChild(save_button_row);

  container.appendChild(table);

}

function add_control_to_table( name, table, denormalisation, normalisation,
  on_change = function(x){},
  min = 0.0, max = 1.0, step = 0.001) {
  
  let row = document.createElement('tr');
  let label = document.createElement('td');
  label.innerHTML = name;

  let input_td = document.createElement('td');
  let input = document.createElement('input');
  input.type = "range";
  input.min = min.toString();
  input.max = max.toString();
  input.step = step.toString();
  input.className = "slider";
  input_td.appendChild(input);

  let output = document.createElement('td');
  output.innerHTML = input.value;

  input.oninput = function() {
    let denorm_value = denormalisation( parseFloat(this.value) )
    on_change( denorm_value );
    output.innerHTML = denorm_value;
  }

  row.appendChild(label);
  row.appendChild(input_td);
  row.appendChild(output);
  table.appendChild(row);

  return {
    'input': input,
    'output': output,
    'norm': normalisation,
    'denorm': denormalisation,
  }
}

function add_colour_control_to_table( name, table, on_change = function(x){}) {
  let row = document.createElement('tr');
  let label = document.createElement('td');
  label.innerHTML = name;

  let input_td = document.createElement('td');
  let input = document.createElement('input');
  input.type = "color";
  input_td.appendChild(input);

  input.oninput = function() {
    on_change( this.value );
  }

  row.appendChild(label);
  row.appendChild(input_td);
  table.appendChild(row);

  return {
    'input': input
  }
}