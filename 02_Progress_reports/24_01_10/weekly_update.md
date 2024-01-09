---
marp: true
theme: custom-dracula
paginate: true
_paginate: false # or use `_paginate: skip`
---

# Agile Trajectory Generation for Tensile Perching with Aerial Robots

---
# Progress Update
### Demonstration Guided Reinforcement Learning with Learned Skills
- Prior Approaches
  - Attempt to follow the demonstrations step by step → Slower Learning
  - New behaviours are not completely unseen - they share subtasks

- Approach
  - Learn a large set of reusable skills from large offline datasets of prior experience collected across many tasks.
  - Skill Based Learning with Demonstrations - an algorithm for demonstration guided RL that levelerages the provided demonstrations by following the demonstrated skills instead of the primitive actions

---
### Forgetful Experience Replay in Hierarchical Reinforcement Learning from Expert Demonstrations
- ForgER algorithm for hierarchical RL, using low-quality demonstrations in complex environments with multiple goals.
- Automatic highlighting of sub-goals in complex tasks improving learning efficiency.
- Control over how "forgetful" the system is. In more unreliable examples the experiences can be

---
### Mapless Navigation for UAVs via Reinforcement Learning from Demonstrations
- Soft Actor Critic from Demonstrations
- Presents an algorithm for Navigation - based on sensor readings from obstacles
- This allowed learning even when obstacles were moved
- Gives background for perch based coordinate system

---
### Inverse Reinforcement Learning Control for Trajectory Tracking of a Multirotor UAV
- Learn the control performance of an expert by imitation demonstrations of a UAV operated by an expert pilot.
- From 7 demonstrations follow a figure of 8 style path.
- Learn the Reward function to achieve this
- Input data in demonstrations is assumed to be optimal 
  - therefore the paper focusses on mimicking rather than improving on the trajectories.

---
# General Plans

---
# Project Plan
- (2 Weeks) - Interim Report (Introduction, Background, Project Plan, Evaluation Plan, Ethical Discussion)


---
# Evaluation Plan

---
# Plans Until Next
- Largely be focussed on the Interim Deadline for the next 2 weeks

---
# Questions
- Background: Level of expectation of understanding of readers?
  - i.e. should I expect understanding of RL?