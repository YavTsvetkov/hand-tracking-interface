"""
Camera module for handling frame capture and preprocessing.
"""

# Import modules with fallback handling
try:
    from .libcamera_capture import LibcameraCapture
except ImportError:
    print("Warning: LibcameraCapture could not be imported")
    
try:
    from .frame_processor import FrameProcessor
except ImportError:
    print("Warning: FrameProcessor could not be imported")

__all__ = ['LibcameraCapture', 'FrameProcessor']
