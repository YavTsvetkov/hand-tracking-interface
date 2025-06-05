"""
Configuration settings for the hand tracking application.
"""

class Settings:
    """Application configuration with default settings."""
    
    # Camera settings
    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = "640x480"
    
    # Detection settings
    DEFAULT_MODEL_PATH = "hand_landmark_lite.tflite"
    DEFAULT_CONFIDENCE_THRESHOLD = 0.2  # Lowered threshold - 0.6 is too high for this model
    DEFAULT_SMOOTHING_FACTOR = 0.4
    DEFAULT_CROP_FACTOR = 0.8
    
    # Tracking settings
    DEFAULT_MAX_JUMP = 150
    DEFAULT_DETECTION_LOSS_FRAMES = 5
    DEFAULT_STABILITY_THRESHOLD = 0.6
    DEFAULT_FALSE_POSITIVE_THRESHOLD = 5
    
    # Performance settings
    DEFAULT_FRAME_SKIP = 1
    
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
