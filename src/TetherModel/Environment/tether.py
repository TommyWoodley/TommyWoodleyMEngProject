import pybullet as p
from typing import List, Any
import numpy as np


class Tether:
    RADIUS = 0.005
    MASS = 0.05

    def __init__(self, length: float, top_position: np.ndarray, physics_client: int, num_segments: int = 20) -> None:
        assert isinstance(length, float), "length must be an instance of float"
        assert isinstance(top_position, np.ndarray), "top_position must be an instance of np.ndarray"
        assert isinstance(physics_client, int), "physics_client must be an instance of int"
        assert isinstance(num_segments, int), "num_segments must be an instance of int"

        self.physics_client = physics_client
        self.length = length
        self.num_segments = num_segments
        self.segment_length = length / num_segments
        self.top_position = top_position
        self.segment_mass = self.MASS  # Distribute the mass across the segments
        self.segments = []

        self._parent_frame_pos = np.array([0, 0, -0.5 * self.segment_length], dtype=np.float32)
        self._child_frame_pos = np.array([0, 0, 0.5 * self.segment_length], dtype=np.float32)
        self._body_centre_top = np.array([0, 0, 0.5 * self.length], dtype=np.float32)
        self._body_centre_bottom = np.array([0, 0, -0.5 * self.length], dtype=np.float32)
        self._object_len = np.array([0, 0, self.length], dtype=np.float32)

        self.create_tether()

        mid_index = len(self.segments) // 2
        self.mid_indices = [mid_index - 1, mid_index]

        self.prev_angle = None
        self.cumulative_angle_change = 0.0
        self.wraps = 0.0

    def create_tether(self) -> None:
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

            p.changeDynamics(segment_id, -1, lateralFriction=1.0)

            # Connect this segment to the previous one (if not the first)
            if i > 0:
                self.create_rotational_joint(
                    parent_body_id=self.segments[i - 1],
                    child_body_id=segment_id,
                    parent_frame_pos=self._parent_frame_pos,
                    child_frame_pos=self._child_frame_pos
                )

    def compute_total_rotation(self):
        pos, _ = p.getBasePositionAndOrientation(self.segments[-1])
        last_x = pos[0]
        last_y = pos[2]
        delta_x = last_x - 0
        delta_y = 2.7 - last_y

        # Compute the angle using arctan2, which considers quadrant location
        angle_radians = np.arctan2(delta_x, delta_y)  # swapped x and y to align with the vertical
        angle_degrees = np.degrees(angle_radians)

        if self.prev_angle is not None:
            # Calculate angle change considering the wrap around at 180/-180
            angle_change = angle_degrees - self.prev_angle
            if angle_change > 180:
                angle_change -= 360
            elif angle_change < -180:
                angle_change += 360

            # Update cumulative angle change
            self.cumulative_angle_change += angle_change

            # Update wraps as a float
            self.wraps = self.cumulative_angle_change / 360.0

        # Update the previous angle for the next call
        self.prev_angle = angle_degrees

        return abs(self.wraps)

    def get_segments(self):
        return self.segments

    def get_mid_point(self):
        positions = [p.getBasePositionAndOrientation(obj_id)[0] for obj_id in
                     (self.segments[i] for i in self.mid_indices)]

        # Calculate the midpoint
        midpoint = [(pos1 + pos2) / 2 for pos1, pos2 in zip(positions[0], positions[1])]
        return midpoint

    def get_world_centre_bottom(self) -> np.ndarray:
        return self.top_position - self._object_len

    def get_body_centre_top(self) -> np.ndarray:
        return self._body_centre_top

    def get_body_centre_bottom(self) -> np.ndarray:
        return self._body_centre_bottom

    def attach_to_drone(self, drone: Any) -> None:
        drone_pos = drone.get_body_centre_bottom()
        tether_attachment_point = self._child_frame_pos
        self.create_fixed_joint(parent_body_id=drone.model,
                                child_body_id=self.segments[0],  # Top segment
                                parent_frame_pos=drone_pos,
                                child_frame_pos=tether_attachment_point)

    def attach_weight(self, weight: Any) -> None:

        # Attach the weight to the bottom segment
        tether_attachment_point = self._parent_frame_pos
        weight_attachment_point = weight.get_body_centre_top()
        self.create_fixed_joint(parent_body_id=self.segments[-1],  # Bottom segment
                                child_body_id=weight.weight_id,
                                parent_frame_pos=tether_attachment_point,
                                child_frame_pos=weight_attachment_point)

    def create_rotational_joint(self, parent_body_id: int, child_body_id: int, parent_frame_pos: np.ndarray,
                                child_frame_pos: np.ndarray) -> None:
        assert isinstance(parent_body_id, int), "parent_body_id must be an instance of int"
        assert isinstance(child_body_id, int), "child_body_id must be an instance of int"
        assert isinstance(parent_frame_pos, np.ndarray), "parent_frame_pos must be an instance of np.ndarray"
        assert isinstance(child_frame_pos, np.ndarray), "child_frame_pos must be an instance of np.ndarray"

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

    def create_fixed_joint(self, parent_body_id: int, child_body_id: int, parent_frame_pos: List[float],
                           child_frame_pos: List[float]) -> None:
        assert isinstance(parent_body_id, int), "parent_body_id must be an instance of int"
        assert isinstance(child_body_id, int), "child_body_id must be an instance of int"
        assert isinstance(parent_frame_pos, (List, np.ndarray)), "wrong type"
        assert isinstance(child_frame_pos, (List, np.ndarray)), "child_frame_pos must be an instance of np.ndarray"

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

    def cancel_gravity(self) -> None:
        for seg in self.segments:
            p.applyExternalForce(seg, -1, [0, 0, 10], [0, 0, 0], p.WORLD_FRAME)