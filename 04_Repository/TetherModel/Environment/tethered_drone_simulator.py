import pybullet as p
import numpy as np
from TetherModel.Environment.drone import Drone
from TetherModel.Environment.tether import Tether
from TetherModel.Environment.weight import Weight
from TetherModel.Environment.environment import Environment


class TetheredDroneSimulator:
    def __init__(self, drone_pos: np.ndarray) -> None:
        assert isinstance(drone_pos, np.ndarray), "drone_pos must be an instance of np.ndarray"

        self.drone_pos = drone_pos
        self.physicsClient = p.connect(p.GUI)
        p.setPhysicsEngineParameter(numSolverIterations=500)
        p.setGravity(0, 0, -10)
        self.drone = Drone(self.drone_pos)
        tether_top_position = self.drone.get_world_centre_bottom()
        self.tether = Tether(length=1.0, top_position=tether_top_position, physics_client=self.physicsClient)
        self.tether.attach_to_drone(drone=self.drone)
        tether_bottom_position = self.tether.get_world_centre_bottom()
        self.weight = Weight(top_position=tether_bottom_position)
        self.tether.attach_weight(weight=self.weight)
        self.environment = Environment()
        self.environment.add_tree_branch([0, 0, 2.7])

    def step(self, action: np.ndarray = None) -> None:
        assert isinstance(action, (np.ndarray, type(None))), "action must be an instance of np.ndarray"

        # Update drone position
        if action is not None:
            self.drone_pos += action
            print(self.drone_pos)
            self.drone.set_position(self.drone_pos)
        self.weight.apply_drag()
        # Step the physics simulation
        p.stepSimulation()

    def reset(self, pos: np.ndarray) -> None:
        assert isinstance(pos, np.ndarray), "pos must be an instance of np.ndarray"

        p.resetSimulation()
        self.drone_pos = pos
        self.drone = Drone(pos)
        tether_top_position = self.drone.get_world_centre_bottom()
        self.tether = Tether(length=1.0, top_position=tether_top_position, physics_client=self.physicsClient)
        self.tether.attach_to_drone(drone=self.drone)
        tether_bottom_position = self.tether.get_world_centre_bottom()
        self.weight = Weight(top_position=tether_bottom_position)
        self.tether.attach_weight(weight=self.weight)
        self.environment = Environment()
        self.environment.add_tree_branch([0, 0, 2.7])

    def close(self) -> None:
        p.disconnect(self.physicsClient)
