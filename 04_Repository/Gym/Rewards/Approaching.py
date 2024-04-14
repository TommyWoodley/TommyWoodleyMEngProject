import math

class CircularApproachingReward():
    def within_sector(self, center_x, center_y, radius, start_angle, end_angle, point_x, point_y):
        # Calculate the distance from the point to the center of the circle
        distance = math.sqrt((point_x - center_x) ** 2 + (point_y - center_y) ** 2)
        
        # Check if the point is within the circle's radius
        if distance > radius:
            return False, 0
        
        # Calculate the angle from the center to the point in radians
        angle_radians = math.atan2(point_y - center_y, point_x - center_x)
        
        # Convert angle to degrees for easier handling, normalizing to [0, 360)
        angle_degrees = math.degrees(angle_radians) % 360
        start_angle %= 360
        end_angle %= 360
        
        # Normalize angles to ensure comparison logic works if sector crosses 360 degrees
        if start_angle > end_angle:
            within = (angle_degrees >= start_angle and angle_degrees <= 360) or (angle_degrees >= 0 and angle_degrees <= end_angle)
        else:
            within = (start_angle <= angle_degrees <= end_angle)
        
        return within, 1 - (distance / radius)
    
    def calc_reward(self, state):
        center_x, center_y = 0, 2.7
        radius = 3
        start_angle, end_angle = 225, 315  # Define the arc from 30 degrees to 150 degrees

        x, _, z = state
        is_within, norm_distance = self.within_sector(center_x, center_y, radius, start_angle, end_angle, x, z)
        if is_within:
            return -5 * norm_distance
        
        center_x, center_y = 0, 3.5
        radius = 3
        start_angle, end_angle = 45, 135
        is_within, norm_distance = self.within_sector(center_x, center_y, radius, start_angle, end_angle, x, z)
        if is_within:
            return -5 * norm_distance
        
        return 0

