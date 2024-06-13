from tethered_drone_simulator import TetheredDroneSimulator
from typing import List
import numpy as np


class TetheredDroneSimulatorRunner:
    def __init__(self, xs: List[float], zs: List[float]) -> None:
        self.prev_pos = np.array([xs[0], 0, zs[0] + 3], dtype=np.float32)
        self.simulator = TetheredDroneSimulator(self.prev_pos)
        self.xs = xs
        self.zs = zs
        self.iteration = 0

    def run(self) -> None:
        already_moved = False
        action_size = None
        action_mags = []
        while True:
            it = min(self.iteration, (len(self.xs) - 1))
            x = self.xs[it]
            z = self.zs[it] + 3
            drone_pos = np.array([x, 0, z], dtype=np.float32)
            self.iteration += 495

            action = drone_pos - self.prev_pos
            action_mags.append(np.linalg.norm(action))
            if self.iteration < len(self.xs) * 2:
                self.simulator.step(action)
            elif not already_moved:
                self.simulator.step(np.array([-0.2, 0, 0], dtype=np.float32))
                already_moved = True
                action_size = np.mean(action_mags)
            else:
                self.simulator.step()
            self.prev_pos = drone_pos
            print("x: ", x, " z: ", z, "action_mag: ", action_size)
