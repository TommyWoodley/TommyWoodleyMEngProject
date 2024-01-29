import pybullet as p

class Tether:
    def __init__(self, length, top_position, radius=0.01, mass=0.01):
        self.length = length
        self.radius = radius
        self.mass = mass
        self.top_position = top_position
        top_x, top_y, top_z = top_position
        self.base_position = [top_x, top_y, top_z - 0.5*length - 0.001]
        self.model = self.create_tether()

    def create_tether(self):
        # Collision shape of the tether (cylinder for simplicity)
        collisionShapeId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.radius, height=self.length)
        visualShapeId = p.createVisualShape(p.GEOM_CYLINDER, radius=self.radius, length=self.length, rgbaColor=[0, 0, 1, 1])

        # Create the tether body
        tether_id = p.createMultiBody(baseMass=self.mass,
                                      baseCollisionShapeIndex=collisionShapeId,
                                      baseVisualShapeIndex=visualShapeId,
                                      basePosition=self.base_position,
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        return tether_id
    
    def get_bottom_pos(self):
        top_x, top_y, top_z = self.top_position
        return [top_x, top_y, top_z - self.length]

    def attach_to_drone(self, drone):
        # Calculate the attachment point on the drone
        drone_pos = [0, 0, -0.025]
        tether_attachment_point = [0, 0, 0.5 * self.length]  # Adjust if needed

        # Create a fixed joint between the drone and the tether
        p.createConstraint(parentBodyUniqueId=drone.model,
                          parentLinkIndex=-1,
                          childBodyUniqueId=self.model,
                          childLinkIndex=-1,
                          jointType=p.JOINT_FIXED,  # Use a fixed joint
                          jointAxis=[0, 0, 0],  # Not used in fixed joints
                          parentFramePosition=drone_pos,
                          childFramePosition=tether_attachment_point,
                          parentFrameOrientation=[0, 0, 0, 1],
                          childFrameOrientation=[0, 0, 0, 1])

    def attach_weight(self, weight):
        # Calculate the attachment point on the weight
        weight_attachment_point = [0, 0, -0.5 * self.length]  # Adjust if needed

        p.createConstraint(parentBodyUniqueId=self.model,
                          parentLinkIndex=-1,
                          childBodyUniqueId=weight.weight_id,
                          childLinkIndex=-1,
                          jointType=p.JOINT_FIXED,  # Use a fixed joint
                          jointAxis=[0, 0, 0],  # Not used in fixed joints
                          parentFramePosition=weight_attachment_point,
                          childFramePosition=[0, 0, 0.1],
                          parentFrameOrientation=[0, 0, 0, 1],
                          childFrameOrientation=[0, 0, 0, 1])
