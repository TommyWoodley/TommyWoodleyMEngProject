import pybullet as p

class Drone:
    def __init__(self):
        self.startPos = [0, 0, 3]
        self.startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.height = 0.05
        self.halfExtents = [0.1, 0.1, self.height]  # half width, length, and height of the drone box
        
        # Create a simple box to represent the drone
        collisionShapeId = p.createCollisionShape(p.GEOM_BOX, halfExtents=self.halfExtents)
        visualShapeId = p.createVisualShape(p.GEOM_BOX, halfExtents=self.halfExtents, rgbaColor=[1, 0, 0, 1])
        mass = 5  # Adjust mass to your preference
        self.model = p.createMultiBody(mass, collisionShapeId, visualShapeId, self.startPos, self.startOrientation)
        
    def apply_controls(self):
        # Define the upward force
        # Note: Adjust the magnitude of the force based on the mass of the drone and the desired acceleration
        upward_force = [0, 0, 60]  # This is an example value, you'll need to tune it based on your drone's weight and desired acceleration
        
        # Apply the force to the drone's base (the main body)
        p.applyExternalForce(objectUniqueId=self.model, 
                             linkIndex=-1, 
                             forceObj=upward_force, 
                             posObj=[0, 0, 0], 
                             flags=p.WORLD_FRAME)
        
    def get_position(self):
        # Get the current position and orientation of the drone
        position, orientation = p.getBasePositionAndOrientation(self.model)
        print("drone pos:", position)
        return position
    
    def get_tether_connection_point(self):
        # Get the current position of the drone
        position, orientation = p.getBasePositionAndOrientation(self.model)
        print("drone pos:", position)
        
        # The tether connection point is at the center bottom of the drone
        # Since it's a box, we subtract half the height from the z coordinate
        tetherConnectionPoint = [position[0] + 0.05, position[1] + 0.05, position[2] - 0.5 * self.height]
        
        return tetherConnectionPoint
