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
        p.setGravity(0, 0, -10)
        self.drone = Drone(self.drone_pos)
        tether_top_position = self.drone.get_world_centre_bottom()
        tether = Tether(length=1.0, top_position=tether_top_position)
        tether.attach_to_drone(drone=self.drone)
        tether_bottom_position = tether.get_world_centre_bottom()
        self.weight = Weight(top_position=tether_bottom_position)
        tether.attach_weight(weight=self.weight)
        self.environment = Environment()

    def step_simulation(self):
        # Step the physics simulation
        p.stepSimulation()
    
    def run(self):
        time.sleep(5)
        while self.iteration < len(self.xs):
            x = self.xs[self.iteration]
            z = self.zs[self.iteration] + 3
            position = [x, 0, z]
            self.iteration += 100
            self.drone.set_position(position)
            self.weight.apply_drag()
            self.step_simulation()
            time.sleep(1./240.)
            print("x: ", x, " z: ", z)
