import pybullet as p

class Tether:
    RADIUS = 0.01
    MASS = 0.01
    def __init__(self, length, top_position, radius=0.01, mass=0.01):
        self.length = length
        self.top_position = top_position
        top_x, top_y, top_z = top_position
        self.base_position = [top_x, top_y, top_z - 0.5 * length]
        self.model = self.create_tether()

    def create_tether(self):
        # Collision shape of the tether (cylinder for simplicity)
        collisionShapeId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.RADIUS, height=self.length)
        visualShapeId = p.createVisualShape(p.GEOM_CYLINDER, radius=self.RADIUS, length=self.length, rgbaColor=[0, 0, 1, 1])

        # Create the tether body
        tether_id = p.createMultiBody(baseMass=self.MASS,
                                      baseCollisionShapeIndex=collisionShapeId,
                                      baseVisualShapeIndex=visualShapeId,
                                      basePosition=self.base_position,
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
        return tether_id
    
    def get_world_centre_bottom(self):
        top_x, top_y, top_z = self.top_position
        return [top_x, top_y, top_z - self.length]
    
    def get_body_centre_top(self):
        return [0, 0, 0.5 * self.length]
    
    def get_body_centre_bottom(self):
        return [0, 0, -0.5 * self.length]

    def attach_to_drone(self, drone):
        # calculate attachement points in body coordinate frame
        drone_pos = drone.get_body_centre_bottom()
        tether_attachment_point = self.get_body_centre_top()

        self.create_rotational_joint(parent_body_id=drone.model,
                          child_body_id=self.model,
                          parent_frame_pos=drone_pos,
                          child_frame_pos=tether_attachment_point)

    def attach_weight(self, weight):
        # Calculate the attachment point on the weight
        tether_attachment_point = self.get_body_centre_bottom()
        weight_attachment_point = weight.get_body_centre_top()

        self.create_fixed_joint(parent_body_id=self.model,
                          child_body_id=weight.weight_id,
                          parent_frame_pos=tether_attachment_point,
                          child_frame_pos=weight_attachment_point)
    
    def create_rotational_joint(self, parent_body_id, child_body_id, parent_frame_pos, child_frame_pos):
        # Use a fixed point between the drone and the tether
        # TODO: Use a more realistic version of the joints
        p.createConstraint(parentBodyUniqueId=parent_body_id,
                           parentLinkIndex=-1,
                           childBodyUniqueId=child_body_id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_POINT2POINT,
                           jointAxis=[0, 0, 0],
                           parentFramePosition=parent_frame_pos,
                           childFramePosition=child_frame_pos,
                           parentFrameOrientation=[0, 0, 0, 1],
                           childFrameOrientation=[0, 0, 0, 1])
        
    def create_fixed_joint(self, parent_body_id, child_body_id, parent_frame_pos, child_frame_pos):
        # Use a fixed point between the drone and the tether
        # TODO: Use a more realistic version of the joints
        p.createConstraint(parentBodyUniqueId=parent_body_id,
                           parentLinkIndex=-1,
                           childBodyUniqueId=child_body_id,
                           childLinkIndex=-1,
                           jointType=p.JOINT_FIXED,
                           jointAxis=[0, 0, 0],
                           parentFramePosition=parent_frame_pos,
                           childFramePosition=child_frame_pos,
                           parentFrameOrientation=[0, 0, 0, 1],
                           childFrameOrientation=[0, 0, 0, 1])
