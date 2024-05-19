---
marp: true
theme: custom-dracula
paginate: true
_paginate: false # or use `_paginate: skip`
---

# Agile Trajectory Generation for Tensile Perching with Aerial Robots
---
# Project Aim
- Design and Implement a Framework capable of Learning Agile Perching Trajectories from a limited number of non-expert demontrations.
  - Aim to generate a set of complete trajectories that are feasible for a drone to complete.
  - Learning can utilise a small number of demonstrations.

  - The agent need to improve upon the demonstrations:
    - This is based on energy effiency and "smoothness" or non-volatility of trajecteries.
    - Robustness to poor demonstration datasets.

  - Compare to previously created trajectories for "smoothness", feasability and energy efficiency.

---
# Evaluation
- Compare with previous demonstration algorithms
- Compare with previously generated trajectories for part 1 of the manoeuvre.
Based on:
  - Speed
  - Smoothness of trajectories
  - Compare based on performance of demonstration data set (30%, 50%, 80%)
  - Learning Speed
- Successful completion of manoeuvre


---
# Experiments
- Demonstration Data will need to be collected.
- Based on lab opportunity - training could either be a mix of real and simulation or purely simulation.
- Final evaluation - hopefully fully displaying the manoeuvre.

---
# Timeline
- Demonstration Data Collection (10th March) (1/6 Weeks)
- Tether/Drone Dynamics Model (25th Feb) (1/4 Weeks)
- Learning from Demonstrations Model (21st April)
- Energy Efficiency (5th May)

---
# Progress Update
(For 1pm meeting)
- More clearly defined the project aim and plan.
  - Internally I've had an interim report deadline - featuring background work, project plan, project evaluation - I can send out a copy of this after the meeting.
- Experimented with physics simulators - Video

- Started exploring some of the algorithms discussed in background research (NACfD, DQfD)

---
# Progress Update
(For 2pm meeting)

- Catching up on other modules after the interim report deadline.

From my aim for last week:
- Setup code repository
  - Done
- Bring in algorithms that have been looked at in background work (DQfD, NACfD, SAC, DQN)
  - Brought in as submodules in repo.
  - Started to reproduce the work (simulation experiments) (DQfD)

---
# Progress Update
(For 2pm meeting)
From my aim for last week:
- Start investigating realistic simulators for tether dynamics (Box2D, PyBullet).
  - PyBullet - Video - set up a simple version.
- Work out how to start getting a demonstration data set.
  - TODO

---
# Plans Until Next
- Mainly focus on the tether dynamics model
  - More realistic incorporating some physics into the model as a first stage of training.
  - A way to improve the model based on demonstration/flight data.


---
# Questions
Collecting Demonstration Data
  - How can we arrange doing this?
  - And any suggestions on timelines/when?