from tethered_drone_simulator import TetheredDroneSimulator
import time


class TetheredDroneSimulatorRunner:
    def __init__(self, xs, zs):
        self.simulator = TetheredDroneSimulator(xs, zs)
        self.xs = xs
        self.zs = zs
        self.iteration = 0

    def run(self):
        time.sleep(5)
        already_moved = False
        while True:
            it = min(self.iteration, (len(self.xs) - 1))
            x = self.xs[it]
            z = self.zs[it] + 3
            drone_pos = [x, 0, z]
            self.iteration += 500
            if self.iteration < len(self.xs) * 2:
                self.simulator.step(drone_pos)
            elif not already_moved:
                self.simulator.step([x - 0.2, 0, z])
                already_moved = True
            else:
                self.simulator.step()
            time.sleep(1./240.)
            print("x: ", x, " z: ", z)