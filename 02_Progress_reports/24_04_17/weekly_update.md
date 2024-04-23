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
  - Demonstrations from both optimised trajectory (Hann) and Previous Work (Fabian).
  - Using these demonstrations in training.

---
# Last Week
![h:500](./last_week/demo_visualisation.png)

---
![h:550](./last_week/simple_position_based.png)

---
![h:550](./last_week/rewards_graph.png)

---
![h:550](./last_week/sample_trajectories.png)

---

# From Previous Week
- Issues faced
  - Resetting - fixed an issue an issue with gravity in resetting.
  - Reward Function 
    - done based on hitting tether to branch - stop condition is when there is a collision between the centre portion of the tether and the branch.
    - collision - penalise contact between drone and branch - large collision penalty.
    - Trajectory Smoothness Term
- Learning from Demonstrations System
- By next week: Approaching stage finished with comparison results between SAC and SACfD.


---
# Reward
![h:500](./rewards/simple_position_based.png)

---
![h:550](./rewards/position_branch_drone_collision.png)

---
![h:550](./rewards/collision_based_branch_env.png)

---
![h:550](./rewards/sector_based_collision_aviodance.png)

---
# Trajectories
![h:500](./sample_trajectories.png)

---
# Comparison between SAC and SACfD
![h:500](./sacVsSacfD/sac_training.png)

---
![h:500](./sacVsSacfD/sacfd_training.png)

---
# Next Steps
- Further Statistics on the training
  - Crashes.
- Smoothness
  - Already a smoothness term which can be seen in easy tasks.
  - Introduce Prioritised Experience Replay to help combat the harder learning portions.
  - Sampling learnable parameter.
- Move onto next stage:
  - Wrapping

---
# Questions
- For the paper:
  - How would you suggest going about cutting down the background/related work section?
