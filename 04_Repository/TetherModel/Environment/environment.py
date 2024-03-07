import pybullet as p
import pybullet_data
import numpy as np


class Environment:
    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = p.loadURDF("plane.urdf")
        # TODO: Add the perching branch

    def add_tree_branch(self, position, length=1.0, radius=0.02, orientation=[np.pi / 2, 0.1, 0]):
        orientation_quat = p.getQuaternionFromEuler(orientation)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=radius,
                                              length=length, rgbaColor=[0.6, 0.32, 0.17, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=radius, height=length)
        tree_branch_id = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=collision_shape_id,
                                           baseVisualShapeIndex=visual_shape_id,
                                           basePosition=position,
                                           baseOrientation=orientation_quat)
        return tree_branch_id
