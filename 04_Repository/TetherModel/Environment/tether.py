import pybullet as p
from typing import List, Any
import numpy as np


class Tether:
    RADIUS = 0.005
    MASS = 0.1

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
        self.create_tether()

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

            # Connect this segment to the previous one (if not the first)
            if i > 0:
                self.create_rotational_joint(
                    parent_body_id=self.segments[i - 1],
                    child_body_id=segment_id,
                    parent_frame_pos=np.array([0, 0, -0.5 * self.segment_length], dtype=np.float32),
                    child_frame_pos=np.array([0, 0, 0.5 * self.segment_length], dtype=np.float32)
                )

    def get_world_centre_bottom(self) -> np.array:
        top_x, top_y, top_z = self.top_position
        return np.array([top_x, top_y, top_z - self.length], dtype=np.float32)

    def get_body_centre_top(self) -> np.array:
        return np.array([0, 0, 0.5 * self.length], dtype=np.float32)

    def get_body_centre_bottom(self) -> np.array:
        return np.array([0, 0, -0.5 * self.length], dtype=np.float32)

    def attach_to_drone(self, drone: Any) -> None:
        drone_pos = drone.get_body_centre_bottom()
        tether_attachment_point = np.array([0, 0, 0.5 * self.segment_length], dtype=np.float32)
        self.create_rotational_joint(parent_body_id=drone.model,
                                     child_body_id=self.segments[0],  # Top segment
                                     parent_frame_pos=drone_pos,
                                     child_frame_pos=tether_attachment_point)

    def attach_weight(self, weight: Any) -> None:

        # Attach the weight to the bottom segment
        tether_attachment_point = np.array([0, 0, -0.5 * self.segment_length], np.float32)
        weight_attachment_point = weight.get_body_centre_top()
        self.create_fixed_joint(parent_body_id=self.segments[-1],  # Bottom segment
                                child_body_id=weight.weight_id,
                                parent_frame_pos=tether_attachment_point,
                                child_frame_pos=weight_attachment_point)

    def create_rotational_joint(self, parent_body_id: int, child_body_id: int, parent_frame_pos: np.ndarray,
                                child_frame_pos: np.ndarray) -> None:
        assert isinstance(parent_body_id, int), "parent_body_id must be an instance of int"
        assert isinstance(child_body_id, int), "child_body_id must be an instance of int"
        assert isinstance(parent_frame_pos, np.ndarray), f"parent_frame_pos must be an instance of np.ndarray, found:{type(parent_frame_pos)}"
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
        assert isinstance(parent_frame_pos, (List, np.ndarray)), f"parent_frame_pos must be an instance of np.ndarray or List, found:{type(parent_frame_pos)}"
        assert isinstance(child_frame_pos, (List, np.ndarray)), "child_frame_pos must be an instance of np.ndarray or List"

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
