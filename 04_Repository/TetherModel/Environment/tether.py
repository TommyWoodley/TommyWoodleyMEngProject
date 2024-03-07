import pybullet as p


class Tether:
    RADIUS = 0.005
    MASS = 0.1

    def __init__(self, length, top_position, physics_client, num_segments=20):
        self.physics_client = physics_client
        self.length = length
        self.num_segments = num_segments
        self.segment_length = length / num_segments
        self.top_position = top_position
        self.segment_mass = self.MASS  # Distribute the mass across the segments
        self.segments = []
        self.create_tether()

    def create_tether(self):
        # Create each segment
        for i in range(self.num_segments):
            segment_top_position = [
                self.top_position[0],
                self.top_position[1],
                self.top_position[2] - i * self.segment_length
            ]
            segment_base_position = [
                segment_top_position[0],
                segment_top_position[1],
                segment_top_position[2] - 0.5 * self.segment_length
            ]

            # Collision and visual shapes
            collisionShapeId = p.createCollisionShape(p.GEOM_CYLINDER, radius=self.RADIUS, height=self.segment_length)
            visualShapeId = p.createVisualShape(p.GEOM_CYLINDER, radius=self.RADIUS,
                                                length=self.segment_length, rgbaColor=[0, 0, 1, 1])

            # Create the segment
            segment_id = p.createMultiBody(baseMass=self.segment_mass,
                                           baseCollisionShapeIndex=collisionShapeId,
                                           baseVisualShapeIndex=visualShapeId,
                                           basePosition=segment_base_position,
                                           baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
            self.segments.append(segment_id)

            # Connect this segment to the previous one (if not the first)
            if i > 0:
                self.create_rotational_joint(
                    parent_body_id=self.segments[i - 1],
                    child_body_id=segment_id,
                    parent_frame_pos=[0, 0, -0.5 * self.segment_length],
                    child_frame_pos=[0, 0, 0.5 * self.segment_length]
                )

    def get_world_centre_bottom(self):
        top_x, top_y, top_z = self.top_position
        return [top_x, top_y, top_z - self.length]

    def get_body_centre_top(self):
        return [0, 0, 0.5 * self.length]

    def get_body_centre_bottom(self):
        return [0, 0, -0.5 * self.length]

    def attach_to_drone(self, drone):
        drone_pos = drone.get_body_centre_bottom()
        tether_attachment_point = [0, 0, 0.5 * self.segment_length]
        self.create_rotational_joint(parent_body_id=drone.model,
                                     child_body_id=self.segments[0],  # Top segment
                                     parent_frame_pos=drone_pos,
                                     child_frame_pos=tether_attachment_point)

    def attach_weight(self, weight):
        # Attach the weight to the bottom segment
        tether_attachment_point = [0, 0, -0.5 * self.segment_length]
        weight_attachment_point = weight.get_body_centre_top()
        self.create_fixed_joint(parent_body_id=self.segments[-1],  # Bottom segment
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

    def cancel_gravity(self):
        for seg in self.segments:
            p.applyExternalForce(seg, -1, [0, 0, 10], [0, 0, 0], p.WORLD_FRAME)
