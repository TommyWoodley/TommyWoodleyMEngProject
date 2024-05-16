---
marp: true
theme: custom-dracula
paginate: true
_paginate: false # or use `_paginate: skip`
---

# Agile Trajectory Generation for Tensile Perching with Aerial Robots

---
# Progress Update
- Demonstrations
  - Waiting on Atar/Kangle for update on demonstrations - I have messaged Atar this morning to see if there's anything that I can help with and update.

---
# From Previously
- Issue around wrapping from different sides.
  - Starting position in state space
  - Previous n states
  - Discussion with Hann
    - Take advantage of the symmetry in the environment.

---
# Symmetry
We can assume symmetry across the x=0 plane i.e. from either approaching side of the branch.

Symmetrical Wrapper
- Takes in the positions and actions and converts the positions to be +ve with respect to the starting position.
- Making the problem easier to solve by only needing to learn from one side.
- Implemented as a Gym Wrapper so that it is easy to add/remove to compare the learning effects.

---
### Symmetry Diagram
TODO

---
# Stages

Approaching
- Speed - visually much shorter and faster trajectories - need to gather some adiditonal data on this for evaluation purposes.

Wrapping
- Waiting
  - Previously using the position of the weight which allowed the network to learn when to move onto the next state.
  - In deployment - This would be complex to actaully keep track of in real life - want to aviod this being part of the state space so that the agent can make decisions without this additional knowledge.
  - Incorporate previous state information - "hovering steps" - keep track of how long the agent has hovered - make decisions based on time in a learned manner.

---
Hanging
- Trajectories with a swinging motion underneath.
  - Questions

---
Sample Approaching Trajectories

---
# Thesis Plan
- Intro
- Background
- Environmental Modelling
  - Initial Environment
  - Wrappers
    - Dimensions
    - Symmetry
    -
- Training

---
# Overall Plan


---
![h:570](./sample_trajectories.png)


---
# Next Steps
- Augment State Space with "directional" knowlege
  - Starting position in state space
  - Previous n states

- Hanging Phase