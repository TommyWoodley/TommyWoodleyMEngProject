import pybullet as p
import time

from drone import Drone
from tether import Tether
from weight import Weight
from environment import Environment


class TetheredDroneSimulator:
    def __init__(self, xs, zs):
        self.xs = xs
        self.zs = zs
        self.drone_pos = [xs[0], 0, zs[0] + 3]
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

    def step(self, action=None):
        # Update drone position
        if action != None:
            self.drone_pos = [self.drone_pos[0] + action[0], self.drone_pos[1] + action[1], self.drone_pos[2] + action[2]]
            self.drone.set_position(self.drone_pos)
        self.weight.apply_drag()
        # Step the physics simulation
        p.stepSimulation()