import pybullet as p

class Weight:
    MASS = 1.0
    RADIUS = 0.1
    def __init__(self, top_position):
        top_x, top_y, top_z = top_position
        self.base_position = [top_x, top_y, top_z - self.RADIUS]
        self.create_weight()

    def create_weight(self):
        collisionShapeId = p.createCollisionShape(p.GEOM_SPHERE, radius=self.RADIUS)
        visualShapeId = p.createVisualShape(p.GEOM_SPHERE, radius=self.RADIUS, rgbaColor=[1, 0, 0, 1])

        self.weight_id = p.createMultiBody(baseMass=self.MASS,
                                           baseCollisionShapeIndex=collisionShapeId,
                                           baseVisualShapeIndex=visualShapeId,
                                           basePosition=self.base_position,
                                           baseOrientation=[0, 0, 0, 1])

    def get_position(self):
        position, _ = p.getBasePositionAndOrientation(self.weight_id)
        return position
