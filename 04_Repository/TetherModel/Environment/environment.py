import pybullet as p
import pybullet_data
import numpy as np
from typing import List


class Environment:
    def __init__(self) -> None:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.ground = p.loadURDF("plane.urdf")
        # TODO: Add the perching branch

    def add_tree_branch(self, position: List[float], length: float = 1.0, radius: float = 0.02,
                        orientation: List[float] = [np.pi / 2, 0.1, 0]) -> None:
        assert isinstance(position, List), "position must be an instance of List"
        assert isinstance(length, float), "child_body_id must be an instance of float"
        assert isinstance(radius, float), f"radius must be an instance of float, found:{type(radius)}"
        assert isinstance(orientation, List), "orientation must be an instance of List"

        orientation_quat = p.getQuaternionFromEuler(orientation)
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_CYLINDER, radius=radius,
                                              length=length, rgbaColor=[0.6, 0.32, 0.17, 1])
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_CYLINDER, radius=radius, height=length)
        tree_branch_id = p.createMultiBody(baseMass=0,
                                           baseCollisionShapeIndex=collision_shape_id,
                                           baseVisualShapeIndex=visual_shape_id,
                                           basePosition=position,
                                           baseOrientation=orientation_quat)
        self.branch_pos = position
        return tree_branch_id
    
    def get_tree_branch_midpoint(self):
        return self.branch_pos
