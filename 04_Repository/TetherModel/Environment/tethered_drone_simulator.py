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

    def step_simulation(self):
        # Step the physics simulation
        p.stepSimulation()

    def run(self):
        time.sleep(5)
        already_moved = False
        while True:
            it = min(self.iteration, (len(self.xs) - 1))
            x = self.xs[it]
            z = self.zs[it] + 3
            position = [x, 0, z]
            self.iteration += 500
            if self.iteration < len(self.xs) * 2:
                self.drone.set_position(position)
            elif not already_moved:
                self.drone.set_position([x - 0.2, 0, z])
                already_moved = True

            self.weight.apply_drag()
            # self.tether.cancel_gravity()
            self.step_simulation()
            time.sleep(1./240.)
            print("x: ", x, " z: ", z)
