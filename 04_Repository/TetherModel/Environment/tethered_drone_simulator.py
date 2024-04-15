import pybullet as p
import numpy as np
import time
from TetherModel.Environment.drone import Drone
from TetherModel.Environment.tether import Tether
from TetherModel.Environment.weight import Weight
from TetherModel.Environment.environment import Environment


class TetheredDroneSimulator:
    def __init__(self, drone_pos: np.ndarray, gui_mode=True) -> None:
        assert isinstance(drone_pos, np.ndarray), "drone_pos must be an instance of np.ndarray"
        self.gui_mode = gui_mode

        self.drone_pos = drone_pos
        if gui_mode:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)
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
        self.branch = self.environment.add_tree_branch([0, 0, 2.7])

    def step(self, action: np.ndarray = None) -> None:
        assert isinstance(action, (np.ndarray, type(None))), "action must be an instance of np.ndarray"

        if self.gui_mode:
            time.sleep(0.001)

        # Update drone position
        if action is not None:
            self.drone_pos += action
            self.drone.set_position(self.drone_pos)
        # Step the physics simulation
        has_collided = self.check_collisions()
        dist_tether_branch = self._distance(self.tether.get_mid_point(), self.environment.get_tree_branch_midpoint())
        dist_drone_branch = self._distance(self.drone.get_world_centre_centre(),
                                           self.environment.get_tree_branch_midpoint())
        p.stepSimulation()
        return has_collided, dist_tether_branch, dist_drone_branch

    def check_collisions(self):
        for part_id in self.tether.get_segments():
            contacts = p.getContactPoints(bodyA=self.branch, bodyB=part_id)
            if contacts:
                return True
        return False

    def reset(self, pos: np.ndarray) -> None:
        assert isinstance(pos, np.ndarray), "pos must be an instance of np.ndarray"

        p.resetSimulation()
        p.setGravity(0, 0, -10)
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

    def _distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))
