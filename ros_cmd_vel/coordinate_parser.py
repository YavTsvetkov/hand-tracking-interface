"""
Coordinate parser for converting palm positions to cmd_vel commands.
"""

import math
import numpy as np


class CoordinateParser:
    """Converts palm pixel coordinates to ROS cmd_vel commands."""
    
    def __init__(self, frame_width=640, frame_height=480, 
                 max_linear_speed=0.5, max_angular_speed=1.0,
                 dead_zone_radius=50, control_zone_radius=150,
                 smoothing_factor=0.2):
        """
        Initialize coordinate parser.
        
        Args:
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels  
            max_linear_speed: Maximum linear velocity (m/s)
            max_angular_speed: Maximum angular velocity (rad/s)
            dead_zone_radius: Dead zone radius in pixels (no movement)
            control_zone_radius: Control zone radius in pixels (max control)
            smoothing_factor: Smoothing factor for commands (0-1)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        self.dead_zone_radius = dead_zone_radius
        self.control_zone_radius = control_zone_radius
        self.smoothing_factor = smoothing_factor
        
        # Frame center
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        
        # Previous command for smoothing
        self.prev_linear_x = 0.0
        self.prev_angular_z = 0.0
        
    def parse_palm_coordinates(self, palm_position):
        """
        Convert palm pixel coordinates to cmd_vel command.
        
        Args:
            palm_position: Tuple of (x, y) pixel coordinates or None
            
        Returns:
            dict: {'linear_x': float, 'angular_z': float}
        """
        if palm_position is None:
            # No hand detected - stop
            cmd_vel = {'linear_x': 0.0, 'angular_z': 0.0}
            self.prev_linear_x = 0.0
            self.prev_angular_z = 0.0
            return cmd_vel
            
        x, y = palm_position
        
        # Calculate distance from center
        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # Check if in dead zone
        if distance < self.dead_zone_radius:
            cmd_vel = {'linear_x': 0.0, 'angular_z': 0.0}
        else:
            # Calculate control intensity (0-1)
            if distance > self.control_zone_radius:
                intensity = 1.0
            else:
                intensity = (distance - self.dead_zone_radius) / (self.control_zone_radius - self.dead_zone_radius)
            
            # Map Y coordinate to linear velocity (forward/backward)
            # Top of frame = forward, bottom = backward
            # Ensure values are clipped to valid range
            linear_factor = np.clip(-dy / (self.frame_height / 2), -1.0, 1.0)  # Normalize to [-1, 1]
            linear_x = linear_factor * intensity * self.max_linear_speed
            
            # Map X coordinate to angular velocity (left/right turn)
            # Left of frame = turn left, right = turn right
            # Ensure values are clipped to valid range
            angular_factor = np.clip(-dx / (self.frame_width / 2), -1.0, 1.0)  # Normalize to [-1, 1] 
            angular_z = angular_factor * intensity * self.max_angular_speed
            
            # Apply smoothing
            linear_x = self.prev_linear_x + self.smoothing_factor * (linear_x - self.prev_linear_x)
            angular_z = self.prev_angular_z + self.smoothing_factor * (angular_z - self.prev_angular_z)
            
            # Store for next iteration
            self.prev_linear_x = linear_x
            self.prev_angular_z = angular_z
            
            cmd_vel = {
                'linear_x': linear_x,
                'angular_z': angular_z
            }
            
        return cmd_vel
    
    def get_control_info(self, palm_position):
        """
        Get debug information about control mapping.
        
        Args:
            palm_position: Tuple of (x, y) pixel coordinates or None
            
        Returns:
            dict: Debug information about control state
        """
        if palm_position is None:
            return {
                'in_dead_zone': False,
                'in_control_zone': False,
                'distance_from_center': 0,
                'control_intensity': 0.0,
                'normalized_dx': 0.0,
                'normalized_dy': 0.0
            }
            
        x, y = palm_position
        dx = x - self.center_x
        dy = y - self.center_y
        distance = math.sqrt(dx*dx + dy*dy)
        
        in_dead_zone = distance < self.dead_zone_radius
        in_control_zone = distance <= self.control_zone_radius
        
        if in_dead_zone:
            intensity = 0.0
        elif distance > self.control_zone_radius:
            intensity = 1.0
        else:
            intensity = (distance - self.dead_zone_radius) / (self.control_zone_radius - self.dead_zone_radius)
            
        return {
            'in_dead_zone': in_dead_zone,
            'in_control_zone': in_control_zone,
            'distance_from_center': distance,
            'control_intensity': intensity,
            'normalized_dx': dx / (self.frame_width / 2),
            'normalized_dy': dy / (self.frame_height / 2),
            'center_x': self.center_x,
            'center_y': self.center_y
        }
    
    def update_parameters(self, **kwargs):
        """Update parser parameters."""
        if 'frame_width' in kwargs:
            self.frame_width = kwargs['frame_width']
            self.center_x = self.frame_width // 2
        if 'frame_height' in kwargs:
            self.frame_height = kwargs['frame_height']
            self.center_y = self.frame_height // 2
        if 'max_linear_speed' in kwargs:
            self.max_linear_speed = kwargs['max_linear_speed']
        if 'max_angular_speed' in kwargs:
            self.max_angular_speed = kwargs['max_angular_speed']
        if 'dead_zone_radius' in kwargs:
            self.dead_zone_radius = kwargs['dead_zone_radius']
        if 'control_zone_radius' in kwargs:
            self.control_zone_radius = kwargs['control_zone_radius']
        if 'smoothing_factor' in kwargs:
            self.smoothing_factor = kwargs['smoothing_factor']
    
    def get_cmd_vel_display_values(self, palm_position):
        """
        Get cmd_vel values mapped to display range [0.0, 10.0] for visualization.
        
        Args:
            palm_position: Tuple of (x, y) pixel coordinates or None
            
        Returns:
            dict: {'linear_display': float, 'angular_display': float}
        """
        cmd_vel = self.parse_palm_coordinates(palm_position)
        
        # Map from [-max_speed, max_speed] to [0.0, 10.0]
        linear_display = ((cmd_vel['linear_x'] / self.max_linear_speed) + 1.0) * 5.0
        angular_display = ((cmd_vel['angular_z'] / self.max_angular_speed) + 1.0) * 5.0
        
        # Clamp to [0.0, 10.0] range
        linear_display = max(0.0, min(10.0, linear_display))
        angular_display = max(0.0, min(10.0, angular_display))
        
        return {
            'linear_display': linear_display,
            'angular_display': angular_display,
            'linear_raw': cmd_vel['linear_x'],
            'angular_raw': cmd_vel['angular_z']
        }
