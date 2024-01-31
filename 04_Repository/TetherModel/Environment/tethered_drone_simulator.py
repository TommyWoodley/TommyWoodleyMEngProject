import pybullet as p
import time

from drone import Drone
from tether import Tether
from weight import Weight
from environment import Environment

class TetheredDroneSimulator:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -1)
        self.drone = Drone()
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
        position_gain = 20
        velocity_gain = 10
        target_position = [0, 1, 3]  # for example
        
        while True:
            # self.drone.apply_controls(upward_force=3)
            self.drone.navigate_to_position(target_position=target_position, position_gain=position_gain, velocity_gain=velocity_gain)
            self.drone.stabilize_orientation()
            self.weight.apply_drag()
            self.step_simulation()
            time.sleep(1./240.)
