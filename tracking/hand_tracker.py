"""
Hand state tracking and filtering.
"""

import time
from collections import deque

class HandTracker:
    """Tracks hand presence and filters false detections."""
    
    def __init__(self, history_size=10, stable_threshold=3, detection_loss_frames=5):
        """Initialize hand tracker with tracking parameters."""
        # Tracking history
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Detection state
        self.is_hand_present = False
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.stable_threshold = stable_threshold  # Frames needed to confirm detection
        self.detection_loss_frames = detection_loss_frames  # Frames before hand considered lost
        
        # Tracking quality
        self.tracking_quality = 0.0  # 0.0-1.0
        
    def update(self, position=None, confidence=None):
        """Update tracker with new detection data."""
        current_time = time.time()
        
        if position is not None:
            # We have a detection
            self.position_history.append(position)
            self.time_history.append(current_time)
            if confidence is not None:
                self.confidence_history.append(confidence)
            
            # Increase detection counter
            self.consecutive_detections += 1
            self.consecutive_non_detections = 0
            
            # Update hand presence status
            if self.consecutive_detections >= self.stable_threshold:
                self.is_hand_present = True
                
            # Update tracking quality
            self._update_tracking_quality()
            
            return True
        else:
            # No detection
            self.consecutive_detections = 0
            self.consecutive_non_detections += 1
            
            # Update hand presence status
            if self.consecutive_non_detections >= self.detection_loss_frames:
                self.is_hand_present = False
                self.tracking_quality = 0.0
                
            return False
    
    def _update_tracking_quality(self):
        """Calculate tracking quality based on detection history."""
        # No data yet
        if not self.position_history or not self.confidence_history:
            self.tracking_quality = 0.0
            return
            
        # Calculate average confidence
        avg_confidence = sum(self.confidence_history) / len(self.confidence_history)
        
        # Consider detection continuity
        detection_factor = min(1.0, self.consecutive_detections / self.stable_threshold)
        
        # Combine factors to get tracking quality
        self.tracking_quality = avg_confidence * 0.7 + detection_factor * 0.3
    
    def get_tracking_quality(self):
        """Return current tracking quality (0.0-1.0)."""
        return self.tracking_quality
    
    def is_tracking_stable(self, threshold=0.6):
        """Check if tracking is stable based on quality threshold."""
        return self.is_hand_present and self.tracking_quality >= threshold
    
    def reset(self):
        """Reset tracker state."""
        self.position_history.clear()
        self.time_history.clear()
        self.confidence_history.clear()
        self.is_hand_present = False
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.tracking_quality = 0.0