"""
Hand state tracking and filtering.
"""

import time
from collections import deque

class HandTracker:
    """Tracks hand presence with balanced filtering for maximum raw accuracy and false positive rejection."""
    
    def __init__(self, history_size=5, stable_threshold=3, detection_loss_frames=2):
        """Initialize hand tracker with stricter detection criteria for false positive rejection."""
        # Tracking history - still using small buffer for responsive tracking
        self.position_history = deque(maxlen=5)  # Only keep 5 frames of history
        self.time_history = deque(maxlen=5)      # Only keep 5 frames of time
        self.confidence_history = deque(maxlen=5) # Only keep 5 frames of confidence
        
        # Detection state - more cautious about accepting new detections
        self.is_hand_present = False
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.stable_threshold = 3       # Require 3 consecutive detections to confirm presence (stronger false positive rejection)
        self.detection_loss_frames = 2  # Still quick to lose detection for responsiveness
        
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
        """Calculate tracking quality based primarily on raw confidence for maximum accuracy."""
        # No data yet
        if not self.position_history or not self.confidence_history:
            self.tracking_quality = 0.0
            return
            
        # Use most recent confidence value for more responsive tracking
        # This prioritizes raw data over historical averages
        latest_confidence = self.confidence_history[-1]
        
        # Only very slightly consider detection continuity
        detection_factor = min(1.0, self.consecutive_detections / self.stable_threshold)
        
        # Heavily weight current confidence (90%) vs detection history (10%)
        # This gives us more responsive, raw tracking data
        self.tracking_quality = latest_confidence * 0.9 + detection_factor * 0.1
    
    def get_tracking_quality(self):
        """Return current tracking quality (0.0-1.0)."""
        return self.tracking_quality
    
    def is_tracking_stable(self, threshold=0.3):
        """Check if tracking is stable with a lower threshold for faster response.
        Optimized for raw position accuracy with minimal filtering."""
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