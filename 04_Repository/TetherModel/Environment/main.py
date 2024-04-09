import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from run_tethered_drone_simulator import TetheredDroneSimulatorRunner
from typing import List
import numpy as np
import sys


def main(xs: List[float], zs: List[float]) -> None:
    simulator = TetheredDroneSimulatorRunner(xs, zs)
    simulator.run()


if __name__ == "__main__":
    sys.path.append('/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository/')
    data_loaded = np.loadtxt("/Users/tomwoodley/Desktop/TommyWoodleyMEngProject/04_Repository"
                             + "/Data/PreviousWorkTrajectories/Original/trajectory_data.txt", delimiter=',')

    # If you need to separate the loaded data back into cycleX, cycleZ, and finalPosDrone
    xs = data_loaded[:-1, 0]  # All rows except the last, first column
    zs = data_loaded[:-1, 1]  # All rows except the last, second column
    main(xs, zs)
