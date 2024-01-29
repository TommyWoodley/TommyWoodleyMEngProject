import pybullet as p
import pybullet_data

class Environment:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = p.loadURDF("plane.urdf")
        # TODO: Add the perching branch
