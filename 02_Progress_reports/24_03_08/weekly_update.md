---
marp: true
theme: custom-dracula
paginate: true
_paginate: false # or use `_paginate: skip`
---

# Agile Trajectory Generation for Tensile Perching with Aerial Robots

---
# Progress Update
Focussed around the learning from previous demos of the approaching stage.
- Adapted the environment to be part of a gym wrapper & standard baselines wrapper.
  - Gym: Framework of RL environments.
  - Standard Baselines: Set of reliable RL algorithms in PyTorch.
- Approaching Stage
  - R = $-distance\_to\_goal$ where goal is just above the branch.
  - Soft Actor Critic (from Demonstrations).

---
- Current Issues
  - Simulator Staying Alive
  - Need to decide what is an appropriate max action for the drone: 0.001m at 240Hz = 0.24m/s
  - Also considering position based instead of action based.
  - Speed
    - ~~Numpy: Inefficient list operations in pybullet simulation (36%)~~
    - Headless: Currently only implemented a "human" env, want a way in the wrapper to not show the GUI. (52%)
    - Imperial DoC - GPUs: Currently just running on my own laptop.
    - Parallelisation: Stable Baselines provides relatviely straightforward methods to use multiple enviornments at the same time - in parallel.

---
# Overall Progess
- Exams for 2 weeks: 
  - Unavailable for 2 weeks: Friday 15th March & Friday 22nd March.
  - Next meeting that I will be here are: 1pm Friday 29th March, 9:30am Friday 5th April.


---
# Demonstrations
- Approaching Stage 5 demonstrations seems to perform quite well.

| Expert Demos | Mixed Demos | Poor Quality | Very Poor Quality |
|----------|----------|----------|----------|
| 5x Fast Demos | 2x Fast Demos | 0x Fast Demos | 0x Fast Demos |
| 0x Slow Manuever | 2x Slow Manuever  | 4x Slow Manuever | 0x Slow Manuever |
| 0x Random Flight | 1x Random Flight | 1x Random Flight | 5x Random Flight |

- Total: 5x Fast Demos, 5x Slow Mauever, 5x Random Flight.

---
# Reward System
- Sparse Reward: Easy to define but harder to train from, Some evidence that can still perform well with demonstrations.
  - R = hanging underneath the branch structure - defining a zone.
- Dense Reward: Harder to define but makes training faster.
  - Approaching
  - Wrapping
  - Hanging
  - Safe Tether
  - Faster Agile (1-shot) motion