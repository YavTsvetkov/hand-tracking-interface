"""
Coordinate parsing module for converting palm center coordinates to cmd_vel commands.
"""

import math
import numpy as np

class CoordinateParser:
    """Parses palm center coordinates and converts them to robot control commands."""
    
    def __init__(self, 
                 frame_width=640, 
                 frame_height=480,
                 max_linear_speed=0.5,
                 max_angular_speed=1.0,
                 dead_zone_radius=50,
                 control_zone_radius=150):
        """
        Initialize coordinate parser.
        
        Args:
            frame_width: Width of the camera frame in pixels
            frame_height: Height of the camera frame in pixels
            max_linear_speed: Maximum linear speed in m/s
            max_angular_speed: Maximum angular speed in rad/s
            dead_zone_radius: Radius of center dead zone in pixels
            control_zone_radius: Radius of control zone in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.dead_zone_radius = dead_zone_radius
        self.control_zone_radius = control_zone_radius
        
        # Calculate frame center
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Smoothing parameters
        self.smoothing_factor = 0.2
        self.last_linear_x = 0.0
        self.last_angular_z = 0.0
        
    def parse_palm_coordinates(self, palm_position):
        """
        Parse palm center coordinates and convert to cmd_vel command.
        
        Args:
            palm_position: Tuple of (x, y) palm center coordinates in pixels
            
        Returns:
            dict: Command velocity with keys 'linear_x', 'angular_z'
        """
        if palm_position is None:
            return {'linear_x': 0.0, 'angular_z': 0.0}
        
        palm_x, palm_y = palm_position
        
        # Calculate distance from center
        dx = palm_x - self.center_x
        dy = palm_y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        
        # Check if within dead zone
        if distance < self.dead_zone_radius:
            return {'linear_x': 0.0, 'angular_z': 0.0}
        
        # Check if within control zone
        if distance > self.control_zone_radius:
            # Normalize to control zone boundary
            dx = dx * (self.control_zone_radius / distance)
            dy = dy * (self.control_zone_radius / distance)
            distance = self.control_zone_radius
        
        # Calculate normalized distances (-1 to 1)
        normalized_dx = dx / self.control_zone_radius
        normalized_dy = dy / self.control_zone_radius
        
        # Map to robot commands
        # Forward/backward: based on Y position (negative Y = forward)
        linear_x = -normalized_dy * self.max_linear_speed
        
        # Left/right turn: based on X position (positive X = right turn)
        angular_z = -normalized_dx * self.max_angular_speed
        
        # Apply smoothing
        linear_x = self._smooth_value(linear_x, self.last_linear_x)
        angular_z = self._smooth_value(angular_z, self.last_angular_z)
        
        # Update last values
        self.last_linear_x = linear_x
        self.last_angular_z = angular_z
        
        return {
            'linear_x': linear_x,
            'angular_z': angular_z
        }
    
    def _smooth_value(self, new_value, last_value):
        """Apply exponential smoothing to a value."""
        return (1 - self.smoothing_factor) * last_value + self.smoothing_factor * new_value
    
    def get_control_info(self, palm_position):
        """
        Get debugging information about the control mapping.
        
        Args:
            palm_position: Tuple of (x, y) palm center coordinates in pixels
            
        Returns:
            dict: Debug information about the control mapping
        """
        if palm_position is None:
            return {
                'distance_from_center': 0,
                'in_dead_zone': True,
                'in_control_zone': False,
                'normalized_dx': 0,
                'normalized_dy': 0
            }
        
        palm_x, palm_y = palm_position
        dx = palm_x - self.center_x
        dy = palm_y - self.center_y
        distance = math.sqrt(dx**2 + dy**2)
        
        return {
            'distance_from_center': distance,
            'in_dead_zone': distance < self.dead_zone_radius,
            'in_control_zone': distance <= self.control_zone_radius,
            'normalized_dx': dx / self.control_zone_radius if distance > 0 else 0,
            'normalized_dy': dy / self.control_zone_radius if distance > 0 else 0,
            'raw_offset': (dx, dy)
        }
    
    def update_parameters(self, **kwargs):
        """Update parser parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Recalculate center if frame size changed
        if 'frame_width' in kwargs or 'frame_height' in kwargs:
            self.center_x = self.frame_width // 2
            self.center_y = self.frame_height // 2
