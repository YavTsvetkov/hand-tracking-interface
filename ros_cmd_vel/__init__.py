"""
ROS 2 cmd_vel module for hand tracking control.
"""

# Always import coordinate parser (no ROS dependency)
from .coordinate_parser import CoordinateParser

# Optional ROS imports - only if ROS is available
_ros_available = False
try:
    from .ros_publisher import RosPublisher, RosManager
    from .integration import HandTrackingRosIntegration
    from config.ros_config import RosConfig
    _ros_available = True
    __all__ = ['CoordinateParser', 'RosPublisher', 'RosManager', 'HandTrackingRosIntegration', 'RosConfig']
except ImportError:
    # ROS not available - only coordinate parser
    __all__ = ['CoordinateParser']
