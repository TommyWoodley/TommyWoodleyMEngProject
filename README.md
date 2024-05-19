# TommyWoodleyMEngProject

## Project Plan

## Structure
This repositroy contains two directories `progress_updates` and `src`:
- `src`- main content of the repository - check out the ReadMe in that folder for more information.
- `progress_updates` - in the process of being migrated to a private repo

### Aim
Design and Implement a Framework for Learning Agile Perching Trajectories from Non-Expert Demonstrations.

Using a small number of demonstrations to perform the required task.
Following demonstrations the agent should improve energy efficiency while still completing the perching task.

### Steps

### 1. Demonstration Data Collection:
   - **Demonstration Acquisition**:
        - Collect a set of trajectories including expert, poor, and incorrect demonstrations.
        - This data will be collected from real-world human-created flights.
      
### 2. Tether Dynamics Model:
   - **Data Generation for Tether Dynamics**:
      - Utilise a physics based simulation engine for modelling tether dynamics using more accurate physics simulations.
      - Create a dataset using physics-based simulations, varying parameters to cover a wide range of scenarios.
   - **Supervised Learning Model**:
      - Design a neural network to predict the position of the weight given previous positions/velocities of the drone.
   - **Separate Training**:
      - Train this model separately to ensure high accuracy and robustness.
      - Validate the model's performance with new simulated data.

### 3. Normalized Actor-Critic from Demonstrations
   - **Algorithm Adaptation**:
     - Adapt a normalized actor-critic method. [1]
   - **Utilization of Demonstrations**:
     - Integrate the provided demonstrations into the learning process to speed up learning.
   - **Reward Mechanism for Task Success**:
     - Design a reward mechanism that uses the critical milestones of the task, such as the number of wraps achieved around the pole and successful disarmament, ensuring these key objectives are directly incentivized. This can be calculated using the tether dynamics model so that it can be used in simulation environments that dont typically support.
   - **Prioritized Replay of Demonstrations**:
     - Improve upon the NAC algorithm to implement a prioritized replay mechanism that selectively revisits important or less frequently seen demonstrations, enhancing the learning efficiency and robustness of the policy.

### 4. Reward Shaping for Energy Efficiency
   - **Thrust Cost-Based Reward Function**:
     - Develop an additional reward component that quantifies the thrust cost, drawing inspiration from relevant literature to create a model that accurately reflects the energy expenditure of different maneuvers.
   - **Balancing Task Achievement and Energy Efficiency**:
     - Experiment with different weighting schemes to balance the original task-related rewards and the energy efficiency rewards, ensuring the agent learns to complete the task effectively while also optimizing for energy conservation.

### 5. Evaluation and Iteration
   - **Simulated Evaluation**:
     - Conduct testing in a simulated environment (such as Gazebo), allowing for safe and efficient iteration.
     - Use simulation to perform extensive hyperparameter tuning and ablation studies, ensuring the robustness and generalizability of the learned policy.
   - **Real-World Testing**:
     - Transition to real-world testing in a controlled environment once the policy demonstrates high proficiency in the simulation.
     - If required utilise learning from the real world with mini-batches of demonstration or gazebo data.
