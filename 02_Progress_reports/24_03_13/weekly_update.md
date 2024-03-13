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
- Action
  - Originally using a single step action system i.e. the drone takes a very small movement action.
  - Adapted to use the 2 level system i.e. waypoints - defined as wrappers
![w:850](systemDesign.drawio-2.png)

---
- Next Steps
  - Starting positions
    - Currently the drone starts in a fixed position unless (this makes the task much easier but unrealistic).
  - Simulator Staying Alive
    - Can't save new trajectories - need some time to debug this.
  - Speed
    - ~~Numpy: Inefficient list operations in pybullet simulation (36%)~~
    - Headless: Currently only implemented a "human" env, want a way in the wrapper to not show the GUI. (52%)
    - Imperial DoC - GPUs: Currently just running on my own laptop.
    - Parallelisation: Stable Baselines provides relatviely straightforward methods to use multiple enviornments at the same time - in parallel.

---
# Overall Progess
- Exams/Break for 2 weeks, I will miss:
  - Wednesday 20th March
  - Friday 22nd March 
  - Wednesday 27th March


---
# Demonstrations
- Approaching Stage 5 demonstrations seems to perform quite well.
- Combination of 3 types of demonstrations:
  - Optimised Trajectory (Hann)
  - Non-Agile Manuever (Fabian/Luca/Maxi)
  - Random Flight - Tethered Flight (Luca/Maxi)

| Demo Type | Expert Demos | Non-Agile | Mixed Demos | Poor Quality | Very Poor Quality |
|---------- |----------|----------|----------|----------|----------|
|Optimised| 5 | 0 | 3 | 0 | 0 |
|Non-Agile| 0 | 5 | 2 | 3 | 0 |
|Random   | 1 | 1 | 1 | 3 | 6 |

---
- Total: 5x Fast Demos, 5x Slow Mauever, 5x Random Flight.
- Fast Demos
  - Hopefully get these from Hann - he said he would add me to his GitHub repo this morning.
- Non-Agile Manuever - human piloted flight from a range of approaching angles e.g. 0, 72, 144, 216, 288 degrees approximately.
  - Full manuever including approaching, wrapping and hanging if possible.


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