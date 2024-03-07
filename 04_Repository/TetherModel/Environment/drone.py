import pybullet as p


class Drone:
    # Half width, Half height, Half length
    WIDTH = 0.1
    LENGTH = 0.1
    HEIGHT = 0.05
    MASS = 1.0

    def __init__(self, start_pos):
        self.startPos = start_pos
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.halfExtents = [self.WIDTH, self.LENGTH, self.HEIGHT]  # half width, length, and height of the drone box

        # The drone is represented by a simple box
        collisionShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.halfExtents)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.halfExtents, rgbaColor=[1, 0, 0, 1])
        mass = self.MASS
        self.model = p.createMultiBody(mass, collisionShapeId, visualShapeId, self.startPos, self.startOrientation)

    def movement(self):
        # Set horizontal velocity for the drone
        horizontal_velocity_x = 1.0  # velocity along the x-axis
        horizontal_velocity_y = 0.0  # velocity along the y-axis
        current_velocity, _ = p.getBaseVelocity(self.model)

        p.resetBaseVelocity(
            objectUniqueId=self.model,
            linearVelocity=[horizontal_velocity_x, horizontal_velocity_y, current_velocity[2]],
            angularVelocity=[0, 0, 0]
        )

    def apply_controls(self, upward_force):
        upward_force = [0, 0, upward_force]

        # Apply the given force upwards to the drone - world coordinate frame
        # i.e. upwards along the z axis
        p.applyExternalForce(objectUniqueId=self.model,
                             linkIndex=-1,
                             forceObj=upward_force,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME)

    def get_world_centre_bottom(self):
        # current position of the drone
        position, orientation = p.getBasePositionAndOrientation(self.model)

        # bottom centre of the drone is the centre along com minus the half height
        return [position[0], position[1], position[2] - self.HEIGHT]

    def get_body_centre_bottom(self):
        return [0, 0, - self.HEIGHT]

    def set_position(self, position):
        # No change in orientation, so retrieve the current orientation
        _, current_orientation = p.getBasePositionAndOrientation(self.model)

        # Set the new position and keep the current orientation
        p.resetBasePositionAndOrientation(self.model, position, current_orientation)