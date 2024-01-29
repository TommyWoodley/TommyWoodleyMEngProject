import pybullet as p
import time

from drone import Drone
from tether import Tether
from weight import Weight
from environment import Environment

class TetheredDroneSimulator:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -10)
        self.drone = Drone()
        tether_top_position = self.drone.get_centre_bottom()
        tether = Tether(length=1.0, top_position=tether_top_position)
        tether.attach_to_drone(drone=self.drone)
        tether_bottom_position = tether.get_bottom_pos()
        weight = Weight(top_position=tether_bottom_position)
        tether.attach_weight(weight=weight)
        self.environment = Environment()

    def step_simulation(self):
        # Step the physics simulation
        p.stepSimulation()
    
    def run(self):
        while True:
            self.drone.apply_controls(upward_force=60)
            self.step_simulation()
            time.sleep(1./240.)
