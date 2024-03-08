from tethered_drone_simulator import TetheredDroneSimulator


class TetheredDroneSimulatorRunner:
    def __init__(self, xs, zs):
        self.prev_pos = [xs[0], 0, zs[0] + 3]
        self.simulator = TetheredDroneSimulator(self.prev_pos)
        self.xs = xs
        self.zs = zs
        self.iteration = 0

    def run(self):
        already_moved = False
        while True:
            it = min(self.iteration, (len(self.xs) - 1))
            x = self.xs[it]
            z = self.zs[it] + 3
            drone_pos = [x, 0, z]
            self.iteration += 500

            action = [drone_pos[0] - self.prev_pos[0], drone_pos[1] - self.prev_pos[1], drone_pos[2] - self.prev_pos[2]]
            if self.iteration < len(self.xs) * 2:
                self.simulator.step(action)
            elif not already_moved:
                self.simulator.step([-0.2, 0, 0])
                already_moved = True
            else:
                self.simulator.step()
            self.prev_pos = drone_pos
            print("x: ", x, " z: ", z)
