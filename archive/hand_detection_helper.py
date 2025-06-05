#!/usr/bin/env python3
"""
Hand detection helper utilities for improving detection quality
Contains advanced hand presence validation and tracking utilities
"""

import numpy as np
import time
from collections import deque

class HandTracker:
    """Advanced hand tracker to filter false detections and track hand presence"""
    
    def __init__(self, history_size=10, stable_threshold=3, fixed_pos_threshold=1.5,
                 motion_threshold=5, max_still_frames=20):
        # Tracking history
        self.position_history = deque(maxlen=history_size)
        self.time_history = deque(maxlen=history_size)
        self.confidence_history = deque(maxlen=history_size)
        
        # Detection state
        self.is_hand_present = False
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.stable_threshold = stable_threshold  # Frames needed to confirm detection
        
        # False positive detection
        self.fixed_position_counter = 0
        self.fixed_pos_threshold = fixed_pos_threshold  # Max allowed pixels for "fixed" position
        self.motion_threshold = motion_threshold  # Min pixels of expected hand motion
        self.still_counter = 0
        self.max_still_frames = max_still_frames  # Max frames before questioning if position is stuck
        
        # Tracking quality
        self.tracking_quality = 0.0  # 0.0-1.0
        
    def update(self, position=None, confidence=None):
        """Update tracker with new detection data"""
        current_time = time.time()
        
        if position is not None:
            # We have a detection
            self.position_history.append(position)
            self.time_history.append(current_time)
            if confidence is not None:
                self.confidence_history.append(confidence)
            
            # Check if position is suspiciously fixed (false positive)
            self._check_fixed_position()
            
            # Increase detection counter
            self.consecutive_detections += 1
            self.consecutive_non_detections = 0
            
            # Update hand presence state
            if self.consecutive_detections >= self.stable_threshold and not self._is_suspicious():
                self.is_hand_present = True
                
            # Calculate tracking quality
            self._update_tracking_quality()
            
            return True
        else:
            # No detection
            self.consecutive_detections = 0
            self.consecutive_non_detections += 1
            
            # Clear fixed position counter as we have no position
            self.fixed_position_counter = 0
            self.still_counter = 0
            
            # Update hand presence state
            if self.consecutive_non_detections >= self.stable_threshold:
                self.is_hand_present = False
                
            # Tracking quality degrades with each missed detection
            self._update_tracking_quality()
            
            return False
    
    def _check_fixed_position(self):
        """Check if the position is suspiciously fixed (sign of false positive)"""
        if len(self.position_history) < 2:
            return
        
        # Calculate movement
        current_pos = self.position_history[-1]
        prev_pos = self.position_history[-2]
        
        # Calculate pixel distance
        distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
        
        # Check if position is too stable (potential false positive)
        if distance < self.fixed_pos_threshold:
            self.fixed_position_counter += 1
            
            # If in same position too long, might be stuck/false positive
            if self.fixed_position_counter > 5:
                # Reduce confidence in tracking
                self.tracking_quality *= 0.9
        else:
            self.fixed_position_counter = 0
            
        # Check for natural hand movement
        # Hands naturally have small movements
        if distance < self.motion_threshold:
            self.still_counter += 1
        else:
            self.still_counter = 0
    
    def _is_suspicious(self):
        """Check if current detection pattern looks like a false positive"""
        # Too fixed position for too long is suspicious
        if self.fixed_position_counter > 8:
            return True
            
        # Too still for too long is suspicious 
        # (real hands have natural tremor/movement)
        if self.still_counter > self.max_still_frames:
            return True
            
        # Perfect/rigid grid positions are suspicious
        if len(self.position_history) > 3:
            # If all recent positions have the exact same remainder when divided by small values
            # it's likely a false positive grid pattern
            recent_pos = list(self.position_history)[-3:]
            x_values = [pos[0] for pos in recent_pos]
            y_values = [pos[1] for pos in recent_pos]
            
            # Check for perfectly aligned x or y values (too perfect to be real)
            x_perfect = all(x == x_values[0] for x in x_values)
            y_perfect = all(y == y_values[0] for y in y_values)
            
            if x_perfect and y_perfect:
                return True
                
        return False
    
    def _update_tracking_quality(self):
        """Update tracking quality score based on detection history"""
        if self.is_hand_present:
            # If hand is present, quality improves with each detection
            if len(self.confidence_history) > 0:
                # Use actual confidence if available
                self.tracking_quality = 0.7 * self.tracking_quality + 0.3 * np.mean(list(self.confidence_history))
            else:
                # Otherwise slowly increase with each frame
                self.tracking_quality = min(1.0, self.tracking_quality + 0.05)
                
            # Reduce quality if suspicious
            if self._is_suspicious():
                self.tracking_quality *= 0.8
        else:
            # If no hand, quality decreases
            self.tracking_quality = max(0.0, self.tracking_quality - 0.1)
    
    def get_estimated_position(self):
        """Get estimated hand position based on history"""
        if len(self.position_history) == 0:
            return None
            
        # Simple: return latest position
        return self.position_history[-1]
    
    def get_tracking_quality(self):
        """Get tracking quality score (0.0-1.0)"""
        return self.tracking_quality
    
    def is_tracking_stable(self):
        """Check if tracking is stable enough to be reliable"""
        return self.tracking_quality > 0.6
    
    def reset(self):
        """Reset tracking state"""
        self.position_history.clear()
        self.time_history.clear()
        self.confidence_history.clear()
        self.is_hand_present = False
        self.consecutive_detections = 0
        self.consecutive_non_detections = 0
        self.fixed_position_counter = 0
        self.still_counter = 0
        self.tracking_quality = 0.0

class PositionValidator:
    """Validate hand positions to reject implausible movements"""
    
    def __init__(self, max_speed=800, min_speed=0.5, width=640, height=480):
        self.max_speed = max_speed  # Max pixels per second
        self.min_speed = min_speed  # Min pixels per second for real motion
        self.width = width
        self.height = height
        self.last_valid_pos = None
        self.last_time = None
        self.stationary_counter = 0
        
    def is_valid(self, position, current_time=None):
        """Check if position is physically plausible compared to history"""
        if position is None:
            return False
            
        # Unpack position
        x, y = position
        
        # Basic bounds check
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        # First position is always valid
        if self.last_valid_pos is None or self.last_time is None:
            self.last_valid_pos = position
            self.last_time = current_time or time.time()
            return True
            
        # Calculate time delta
        current_time = current_time or time.time()
        time_delta = current_time - self.last_time
        
        # Prevent division by zero
        if time_delta <= 0:
            time_delta = 0.001
            
        # Calculate speed
        prev_x, prev_y = self.last_valid_pos
        distance = np.sqrt((x - prev_x)**2 + (y - prev_y)**2)
        speed = distance / time_delta  # pixels per second
        
        # Check if speed is physically plausible
        if speed > self.max_speed:
            # Too fast - reject
            return False
            
        # Update history with valid position
        self.last_valid_pos = position
        self.last_time = current_time
        
        # Check for motion/stillness
        if speed < self.min_speed:
            self.stationary_counter += 1
        else:
            self.stationary_counter = 0
            
        return True
        
    def is_suspiciously_still(self, threshold=15):
        """Check if position hasn't changed meaningfully for many frames"""
        return self.stationary_counter > threshold
        
    def reset(self):
        """Reset validation state"""
        self.last_valid_pos = None
        self.last_time = None
        self.stationary_counter = 0


# Testing code
if __name__ == "__main__":
    # Test the hand tracker
    tracker = HandTracker()
    
    # Simulate a series of detections
    for i in range(10):
        # Realistic hand movements
        x = 100 + i * 5 + np.random.normal(0, 2)
        y = 100 + i * 2 + np.random.normal(0, 2)
        
        tracker.update((x, y), 0.8)
        print(f"Update {i}: Hand present: {tracker.is_hand_present}, "
              f"Quality: {tracker.get_tracking_quality():.2f}, "
              f"Suspicious: {tracker._is_suspicious()}")
    
    # Simulate fixed position (false positive)
    for i in range(15):
        # Fixed position - suspicious!
        tracker.update((200, 200), 0.7)
        print(f"Fixed {i}: Hand present: {tracker.is_hand_present}, "
              f"Quality: {tracker.get_tracking_quality():.2f}, "
              f"Suspicious: {tracker._is_suspicious()}")
    
    # Simulate no detections
    for i in range(5):
        tracker.update(None, None)
        print(f"None {i}: Hand present: {tracker.is_hand_present}, "
              f"Quality: {tracker.get_tracking_quality():.2f}")
