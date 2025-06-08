"""
Configuration settings for the hand tracking application.
"""

class Settings:
    """Application configuration with default settings."""
    
    # Camera settings
    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = "640x480"
    
    # Detection settings
    DEFAULT_MODEL_PATH = "models/palm_detection.tflite"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.70  # Significantly increased threshold to strongly filter out false positives
    DEFAULT_SMOOTHING_FACTOR = 0.0       # No explicit smoothing for maximum raw accuracy
    DEFAULT_CROP_FACTOR = 1.0            # No crop for maximum field of view
    
    # Tracking settings - optimized for raw position accuracy
    DEFAULT_MAX_JUMP = 250               # Very high value to accept fast movements
    DEFAULT_DETECTION_LOSS_FRAMES = 2    # Quick to lose detection for more responsive tracking
    DEFAULT_STABILITY_THRESHOLD = 0.3    # Lower stability threshold for faster response
    DEFAULT_FALSE_POSITIVE_THRESHOLD = 8 # Higher threshold to reduce false positive filtering
    
    # Performance settings
    DEFAULT_FRAME_SKIP = 1
    
    # ROS 2 settings
    DEFAULT_ROS_TOPIC = "/cmd_vel"
    DEFAULT_ROS_NODE_NAME = "hand_tracking_controller"
    DEFAULT_ROS_PUBLISH_RATE = 10.0  # Hz
    
    @classmethod
    def get_resolution_as_tuple(cls, resolution_str=None):
        """Convert resolution string to width, height tuple."""
        if resolution_str is None:
            resolution_str = cls.DEFAULT_RESOLUTION
            
        try:
            width, height = resolution_str.split('x')
            return int(width), int(height)
        except (ValueError, AttributeError):
            # Default to 640x480 if parsing fails
            return 640, 480
