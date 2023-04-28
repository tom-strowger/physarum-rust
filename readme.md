


Data:
- Agents (positions + headings).  Indexed
- Chemo 2d texture

Stages

1. Draw positions - render pipeline
Input: Agents
Output: new dots

1. Deposit - render pipeline
Input: new dots
Input: Chemo
Output: Chemo

after, swap Chemo A & B

2. Diffuse - render pipeline
Input: Chemo
Output: Chemo

after, swap Chemo A & B

3. Update agents - compute pipeline
Input: Agents
Input: Chemo
Output: Agents

after, swap Agents A & B