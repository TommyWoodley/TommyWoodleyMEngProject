import pybullet as p
import pybullet_data

class Environment:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = p.loadURDF("plane.urdf")
        # Load other static or dynamic objects into the environment
        
    def add_obstacle(self, obstacle):
        # Add an obstacle to the environment
        pass
