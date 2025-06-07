"""
Configuration settings for ROS 2 cmd_vel control.
"""

class RosConfig:
    """Configuration constants for ROS 2 cmd_vel control."""
    
    # ROS 2 settings
    NODE_NAME = 'hand_tracking_controller'
    TOPIC_NAME = '/cmd_vel'
    PUBLISH_RATE = 10.0  # Hz
    
    # Safety settings
    SAFETY_TIMEOUT = 1.0  # seconds
    
    # Control mapping settings
    MAX_LINEAR_SPEED = 0.5  # m/s
    MAX_ANGULAR_SPEED = 1.0  # rad/s
    
    # Control zones (in pixels)
    DEAD_ZONE_RADIUS = 50    # No movement in this zone
    CONTROL_ZONE_RADIUS = 150  # Maximum control range
    
    # Smoothing
    SMOOTHING_FACTOR = 0.2  # Lower = more smoothing
    
    @classmethod
    def get_parser_config(cls, frame_width=640, frame_height=480):
        """Get configuration dictionary for CoordinateParser."""
        return {
            'frame_width': frame_width,
            'frame_height': frame_height,
            'max_linear_speed': cls.MAX_LINEAR_SPEED,
            'max_angular_speed': cls.MAX_ANGULAR_SPEED,
            'dead_zone_radius': cls.DEAD_ZONE_RADIUS,
            'control_zone_radius': cls.CONTROL_ZONE_RADIUS
        }
    
    @classmethod
    def get_ros_config(cls):
        """Get configuration dictionary for ROS manager."""
        return {
            'node_name': cls.NODE_NAME,
            'topic_name': cls.TOPIC_NAME,
            'publish_rate': cls.PUBLISH_RATE
        }
