"""
Position validation for tracking reliability.
"""

import time
import math
from collections import deque

class PositionValidator:
    """Validates hand position data to detect false positives and physically impossible movements."""
    
    def __init__(self, max_speed=800, min_speed=0.5, width=640, height=480):
        """Initialize position validator with movement constraints."""
        # Movement constraints
        self.max_speed = max_speed  # Maximum pixels per second (physically possible)
        self.min_speed = min_speed  # Minimum pixels per second (for detecting stuck positions)
        self.width = width  # Frame width
        self.height = height  # Frame height
        
        # Position history
        self.position_history = deque(maxlen=10)
        self.time_history = deque(maxlen=10)
        self.still_counter = 0
        
    def is_valid(self, position, current_time=None):
        """Validate if position is physically possible based on movement constraints."""
        if position is None:
            return False
            
        if current_time is None:
            current_time = time.time()
            
        # Check if position is within frame boundaries (allow for some margin outside frame)
        x, y = position
        margin = 50  # Allow detection up to 50px outside frame
        if not (-margin <= x < self.width + margin and -margin <= y < self.height + margin):
            print(f"[DEBUG] Position outside frame: {position}, frame: {self.width}x{self.height}")
            return False
            
        # First position is always valid
        if not self.position_history:
            self.position_history.append(position)
            self.time_history.append(current_time)
            return True
            
        # Calculate movement speed
        prev_position = self.position_history[-1]
        prev_time = self.time_history[-1]
        
        # Avoid division by zero
        time_delta = max(0.001, current_time - prev_time)
        
        dx = position[0] - prev_position[0]
        dy = position[1] - prev_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        speed = distance / time_delta
        
        # Check if movement is within physical limits (be more lenient for first few frames)
        if len(self.position_history) < 3:
            is_valid = True  # Always accept first few positions to establish tracking
        else:
            is_valid = speed <= self.max_speed
            if not is_valid:
                print(f"[DEBUG] Speed too high: {speed} > {self.max_speed}")
        
        # Update history
        self.position_history.append(position)
        self.time_history.append(current_time)
        
        # Check for suspiciously still position
        if distance < self.min_speed:
            self.still_counter += 1
        else:
            self.still_counter = 0
            
        return is_valid
        
    def is_suspiciously_still(self, threshold=15):
        """Check if position hasn't changed significantly for many frames."""
        return self.still_counter >= threshold
        
    def reset(self):
        """Reset validator state."""
        self.position_history.clear()
        self.time_history.clear()
        self.still_counter = 0