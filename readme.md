
# Physarum simulation

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