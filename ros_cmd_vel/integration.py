"""
Hand tracking to ROS 2 cmd_vel integration module.
"""

import time
from .coordinate_parser import CoordinateParser
from .ros_publisher import RosManager
from config.ros_config import RosConfig

class HandTrackingRosIntegration:
    """Integrates hand tracking with ROS 2 cmd_vel commands."""
    
    def __init__(self, frame_width=640, frame_height=480, enable_ros=True):
        """
        Initialize the integration system.
        
        Args:
            frame_width: Camera frame width in pixels
            frame_height: Camera frame height in pixels
            enable_ros: Whether to enable ROS 2 publishing
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.enable_ros = enable_ros
        
        # Initialize coordinate parser
        parser_config = RosConfig.get_parser_config(frame_width, frame_height)
        self.coordinate_parser = CoordinateParser(**parser_config)
        
        # Initialize ROS manager if enabled
        self.ros_manager = None
        if enable_ros:
            try:
                ros_config = RosConfig.get_ros_config()
                self.ros_manager = RosManager(**ros_config)
                self.ros_manager.start()
                print("[INFO] ROS 2 cmd_vel integration started")
            except Exception as e:
                print(f"[WARNING] Failed to start ROS 2: {e}")
                self.enable_ros = False
        
        # Statistics
        self.command_count = 0
        self.last_command_time = 0
        
    def process_palm_position(self, palm_position, tracking_quality=1.0):
        """
        Process palm position and send cmd_vel command.
        
        Args:
            palm_position: Tuple of (x, y) palm center coordinates in pixels
            tracking_quality: Quality of tracking (0.0 to 1.0)
            
        Returns:
            dict: Command velocity and debug information
        """
        # Parse coordinates to cmd_vel
        cmd_vel = self.coordinate_parser.parse_palm_coordinates(palm_position)
        
        # Apply tracking quality factor
        if tracking_quality < 0.5:
            # Reduce command strength for low quality tracking
            cmd_vel['linear_x'] *= tracking_quality
            cmd_vel['angular_z'] *= tracking_quality
        
        # Send to ROS if enabled
        if self.enable_ros and self.ros_manager:
            self.ros_manager.update_command_from_dict(cmd_vel)
        
        # Update statistics
        self.command_count += 1
        self.last_command_time = time.time()
        
        # Get debug information
        control_info = self.coordinate_parser.get_control_info(palm_position)
        
        return {
            'cmd_vel': cmd_vel,
            'control_info': control_info,
            'tracking_quality': tracking_quality,
            'ros_enabled': self.enable_ros
        }
    
    def emergency_stop(self):
        """Send emergency stop command."""
        if self.enable_ros and self.ros_manager:
            self.ros_manager.stop_robot()
        print("[INFO] Emergency stop commanded")
    
    def get_status(self):
        """Get current status of the integration system."""
        current_cmd = {'linear_x': 0.0, 'angular_z': 0.0, 'timestamp': 0.0}
        if self.enable_ros and self.ros_manager:
            current_cmd = self.ros_manager.get_current_command()
        
        return {
            'ros_enabled': self.enable_ros,
            'command_count': self.command_count,
            'last_command_time': self.last_command_time,
            'current_command': current_cmd,
            'frame_size': (self.frame_width, self.frame_height),
            'control_zones': {
                'dead_zone': self.coordinate_parser.dead_zone_radius,
                'control_zone': self.coordinate_parser.control_zone_radius
            }
        }
    
    def update_parameters(self, **kwargs):
        """Update system parameters."""
        # Update coordinate parser
        if any(key in kwargs for key in ['max_linear_speed', 'max_angular_speed', 
                                        'dead_zone_radius', 'control_zone_radius',
                                        'smoothing_factor']):
            self.coordinate_parser.update_parameters(**kwargs)
        
        # Update frame size if changed
        if 'frame_width' in kwargs or 'frame_height' in kwargs:
            self.frame_width = kwargs.get('frame_width', self.frame_width)
            self.frame_height = kwargs.get('frame_height', self.frame_height)
            self.coordinate_parser.update_parameters(
                frame_width=self.frame_width,
                frame_height=self.frame_height
            )
    
    def cleanup(self):
        """Clean up resources."""
        if self.ros_manager:
            self.ros_manager.stop()
        print("[INFO] Hand tracking ROS integration cleaned up")
