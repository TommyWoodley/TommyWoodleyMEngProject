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
        self.iteration = 0

        self.drone_pos = [xs[0], 0, zs[0] + 3]
        self.physicsClient = p.connect(p.GUI)
        p.setPhysicsEngineParameter(numSolverIterations=1000)
        p.setGravity(0, 0, -10)
        self.drone = Drone(self.drone_pos)
        tether_top_position = self.drone.get_world_centre_bottom()
        tether = Tether(length=0.7, top_position=tether_top_position, physics_client=self.physicsClient)
        tether.attach_to_drone(drone=self.drone)
        tether_bottom_position = tether.get_world_centre_bottom()
        self.weight = Weight(top_position=tether_bottom_position)
        tether.attach_weight(weight=self.weight)
        self.environment = Environment()
        self.environment.add_tree_branch([0, 0, 2.9])

    def step_simulation(self):
        # Step the physics simulation
        p.stepSimulation()
    
    def run(self):
        time.sleep(5)
        while True:
            if self.iteration >= len(self.xs):
                self.iteration = len(self.xs) - 1
            x = self.xs[self.iteration]
            z = self.zs[self.iteration] + 3
            position = [x, 0, z]
            self.iteration += 500
            self.drone.set_position(position)
            self.weight.apply_drag()
            self.step_simulation()
            time.sleep(1./240.)
            print("x: ", x, " z: ", z)
