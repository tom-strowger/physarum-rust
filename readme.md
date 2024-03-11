
# Physarum simulation

## Prerequisites
- Rust is installed and `cargo` is available in the path
  - wasm-pack is install via cargo (works for Mac ARM vs npm)
    `cargo install wasm-pack`
- `npm` is installed
- The modules are installed `npm -i webpack-cli`

## Building and running
- `npm run serve`
- Open <localhost::8080> in the browser, optionally specifying parameters
  - `width` int
  - `height` int
  - `step_size` float
  - `rotate_angle` float
  - `sense_angle` float
  - `sense_offset` float
  - `num_agents` int
  - `fg_colour` string e.g. "EEFF89"
  - `bg_colour` string e.g. "4599AA"
  - `decay` float
  - `deposit` float
  - `running` bool

e.g. <localhost:8080?fg_colour=EEFF89&bg_colour=4599AA&width=400&height=400>

## Data:
- Agents (positions + headings).  Access by index.  Double-buffered
- Chemo 2d texture.  Accessed by x,y position. Double-buffered
- New dots 2d texture representing the agent positions with a dot drawn at each.

## Stages

1. Draw positions - render pipeline
   - Input: Agents[A]
   - Output: new dots 2d texture

2. Deposit - compute pipeline
   - Input: new dots
   - Input: Chemo[0]
   - Output: Chemo[1]

3. Diffuse - compute pipeline
   - Input: Chemo[1]
   - Output: Chemo[0]

4. Update agents - compute pipeline
   - Input: Agents[A]
   - Input: Chemo[0]
   - Output: Agents[B]

swap A & B each frame


## Todo:
- Add an input 2d texture which can be used to weight towards regions etc. Ideas:
  - A layer which modifies the decay (e.g. for topology)
  - A layer which has adds a constant chemo to an area (cornflake)
  - A blocking layer which prevents any chemo in an area (no-go)
- Have groups within the population and different attractant areas/points for each.