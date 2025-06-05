"""
Frame format conversion and preprocessing utilities.
"""

import cv2
import numpy as np

class FrameProcessor:
    """Handles frame preprocessing for hand detection."""
    
    def __init__(self, crop_factor=0.8):
        """Initialize frame processor with crop factor."""
        self.crop_factor = crop_factor
    
    def preprocess(self, frame):
        """Preprocess frame for model inference."""
        # Apply center crop if configured
        if self.crop_factor > 0:
            frame = self._center_crop(frame, self.crop_factor)
            
        # Convert to RGB (MediaPipe models expect RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    def _center_crop(self, frame, crop_factor):
        """Apply center crop to focus on important areas."""
        if crop_factor >= 1.0 or crop_factor <= 0:
            return frame
            
        h, w = frame.shape[:2]
        crop_h = int(h * crop_factor)
        crop_w = int(w * crop_factor)
        
        # Calculate crop coordinates to center the crop
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        
        # Apply crop
        return frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
