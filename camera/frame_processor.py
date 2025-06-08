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
        """Preprocess frame for model inference.
        
        Returns:
            tuple: (processed_frame, crop_offset_x, crop_offset_y)
                   crop_offset_x and crop_offset_y are 0 if no crop is applied.
        """
        crop_offset_x, crop_offset_y = 0, 0
        # Apply center crop if configured
        if self.crop_factor > 0 and self.crop_factor < 1.0: # Ensure crop_factor is valid for cropping
            frame, crop_offset_x, crop_offset_y = self._center_crop(frame, self.crop_factor)
            
        # Convert to RGB (MediaPipe models expect RGB)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), crop_offset_x, crop_offset_y
    
    def _center_crop(self, frame, crop_factor):
        """Apply center crop to focus on important areas.
        
        Returns:
            tuple: (cropped_frame, start_x, start_y)
        """
        if crop_factor >= 1.0 or crop_factor <= 0:
            return frame, 0, 0
            
        h, w = frame.shape[:2]
        crop_h = int(h * crop_factor)
        crop_w = int(w * crop_factor)
        
        # Calculate crop coordinates to center the crop
        start_x = (w - crop_w) // 2
        start_y = (h - crop_h) // 2
        
        # Apply crop
        cropped_frame = frame[start_y:start_y+crop_h, start_x:start_x+crop_w]
        return cropped_frame, start_x, start_y
