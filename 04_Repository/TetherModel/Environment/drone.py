import pybullet as p

class Drone:
    # Half width, Half height, Half length
    WIDTH = 0.1
    LENGTH = 0.1
    HEIGHT = 0.05
    MASS = 5
    def __init__(self):
        self.startPos = [0, 0, 3]
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
    
    def navigate_to_position(self, target_position, position_gain, velocity_gain):
        # Get the current state of the drone
        current_position, current_orientation_quaternion = p.getBasePositionAndOrientation(self.model)
        current_velocity, current_angular_velocity = p.getBaseVelocity(self.model)

        # Calculate the position error
        position_error = [target - current for target, current in zip(target_position, current_position)]

        # Desired velocity - could be a simple proportional term or something more complex
        desired_velocity = [position_gain * error for error in position_error]

        # Calculate the velocity error
        velocity_error = [desired - current for desired, current in zip(desired_velocity, current_velocity)]

        # Calculate the corrective force
        corrective_force = [position_gain * pos_err + velocity_gain * vel_err for pos_err, vel_err in zip(position_error, velocity_error)]

        # Apply the corrective force to the drone
        p.applyExternalForce(objectUniqueId=self.model,
                             linkIndex=-1, 
                             forceObj=corrective_force, 
                             posObj=[0, 0, 0],  # apply at center of mass
                             flags=p.WORLD_FRAME)

    def stabilize_orientation(self):
        upright_orientation_euler = [0, 0, 0]  # Upright orientation in Euler angles
        upright_orientation_quaternion = p.getQuaternionFromEuler(upright_orientation_euler)
        current_position, _ = p.getBasePositionAndOrientation(self.model)
    
        # Set the drone's orientation to the desired upright orientation
        p.resetBasePositionAndOrientation(self.model, current_position, upright_orientation_quaternion)