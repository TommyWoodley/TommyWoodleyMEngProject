import pybullet as p
from typing import List
import numpy as np


class Weight:
    MASS: float = 0.5
    RADIUS: float = 0.05
    DRAG_COEF: float = 0.472

    _body_centre_top = np.array([0, 0, RADIUS], dtype=np.float32)

    def __init__(self, top_position: np.ndarray) -> None:
        assert isinstance(top_position, np.ndarray), f"top_position must be an instance of np.ndarray"

        top_x, top_y, top_z = top_position
        self.base_position = [top_x, top_y, top_z - self.RADIUS]
        self.create_weight()
        self.cross_area = 3 * self.RADIUS * self.RADIUS

    def create_weight(self) -> None:
        collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=self.RADIUS)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=self.RADIUS, rgbaColor=[1, 0, 0, 1])

        self.weight_id = p.createMultiBody(baseMass=self.MASS,
                                           baseCollisionShapeIndex=collisionShapeId,
                                           baseVisualShapeIndex=visualShapeId,
                                           basePosition=self.base_position,
                                           baseOrientation=[0, 0, 0, 1])

    def get_position(self) -> List[float]:
        position, _ = p.getBasePositionAndOrientation(self.weight_id)
        return position

    def get_body_centre_top(self) -> np.ndarray:
        return self._body_centre_top

    def apply_drag(self, fluid_density: float = 1.225) -> None:
        velocity, _ = p.getBaseVelocity(self.weight_id)
        speed = (velocity[0]**2 + velocity[1]**2 + velocity[2]**2)**0.5

        if speed == 0:
            return

        drag_force_magnitude = 0.5 * fluid_density * speed**2 * self.DRAG_COEF * self.cross_area
        drag_force_direction = [-velocity[0] / speed, -velocity[1] / speed, -velocity[2] / speed]

        drag_force = [drag_force_direction[i] * drag_force_magnitude for i in range(3)]
        p.applyExternalForce(self.weight_id, -1, drag_force, [0, 0, 0], p.WORLD_FRAME)
