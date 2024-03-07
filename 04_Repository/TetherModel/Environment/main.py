from tethered_drone_simulator import TetheredDroneSimulator

import numpy as np


def main(xs, zs):
    simulator = TetheredDroneSimulator(xs, zs)
    simulator.run()


if __name__ == "__main__":
    data_loaded = np.loadtxt("/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository" +
                             "/Data/PreviousWorkTrajectories/Original/trajectory_data.txt", delimiter=',')

    # If you need to separate the loaded data back into cycleX, cycleZ, and finalPosDrone
    xs = data_loaded[:-1, 0]  # All rows except the last, first column
    zs = data_loaded[:-1, 1]  # All rows except the last, second column
    main(xs, zs)
    