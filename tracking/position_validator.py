"""
Position validation for tracking reliability.
"""

import time
import math
from collections import deque

class PositionValidator:
    """Validates hand position data with ultra-minimal constraints for maximum raw accuracy."""
    
    def __init__(self, max_speed=1500, min_speed=0, width=640, height=480):
        """Initialize position validator with movement constraints to filter false positives."""
        # More realistic movement constraints to detect false positives while accepting true hand movements
        self.max_speed = max_speed * 1.5  # Lowered from 3x to 1.5x to catch unrealistic jumps
        self.min_speed = 0              # Still accept completely stationary positions
        self.width = width              # Frame width
        self.height = height            # Frame height
        
        # Position history - track more context for better false positive detection
        self.position_history = deque(maxlen=15)
        self.time_history = deque(maxlen=15)
        self.still_counter = 0
        self.position_variance = deque(maxlen=5)  # Track position variance to detect artificial patterns
        
    def is_valid(self, position, current_time=None):
        """Enhanced position validation with stronger false positive rejection.
        Applies multiple heuristics to identify and reject false detections."""
        if position is None:
            return False
            
        if current_time is None:
            current_time = time.time()
            
        # Position bounds check with tighter margins
        # Further restricted to reject more false positives
        x, y = position
        margin = 50  # Reduced margin - still allows some off-frame detection but rejects more outliers
            
        if not (-margin <= x < self.width + margin and -margin <= y < self.height + margin):
            print(f"[DEBUG] Position outside frame margins: {position}, frame: {self.width}x{self.height}")
            return False
        
        # Check for sudden large position jumps if we have history
        # Enhanced to catch more types of false positives
        if len(self.position_history) >= 1:
            prev_position = self.position_history[-1]
            dx = position[0] - prev_position[0]
            dy = position[1] - prev_position[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Reject extremely large jumps as false positives
            # Lower threshold to catch more impossible movements
            if distance > self.max_speed:
                print(f"[DEBUG] Rejecting suspicious jump: {distance:.1f} px (likely false positive)")
                return False
                
            # Check for repeated identical jumps (unnatural pattern)
            if len(self.position_history) >= 3:
                prev_prev_position = self.position_history[-2]
                prev_dx = prev_position[0] - prev_prev_position[0]
                prev_dy = prev_position[1] - prev_prev_position[1]
                
                # If two consecutive jumps are identical to 2 decimal places, that's suspicious
                if abs(dx - prev_dx) < 0.01 and abs(dy - prev_dy) < 0.01 and distance > 1.0:
                    print(f"[DEBUG] Rejecting suspicious identical jumps pattern")
                    return False
            
        # First position is always valid
        if not self.position_history:
            self.position_history.append(position)
            self.time_history.append(current_time)
            return True
            
        # For maximum accuracy, accept all raw position data
        # Just calculate speed for logging purposes only
        prev_position = self.position_history[-1]
        prev_time = self.time_history[-1]
        
        # Avoid division by zero
        time_delta = max(0.001, current_time - prev_time)
        
        dx = position[0] - prev_position[0]
        dy = position[1] - prev_position[1]
        distance = math.sqrt(dx*dx + dy*dy)
        speed = distance / time_delta
        
        # Just log very large position changes but accept them all
        if speed > self.max_speed * 1.5:
            print(f"[DEBUG] Note: High speed detected: {speed:.1f} px/s")
            
        # Always return true (valid) for maximum raw accuracy
        is_valid = True
        
        # Update history
        self.position_history.append(position)
        self.time_history.append(current_time)
        
        # For raw position tracking, we don't consider stillness a problem
        # Reset the still counter since we're not using it
        self.still_counter = 0
            
        return is_valid
        
    def is_suspiciously_still(self, threshold=10):
        """Enhanced detection of false positives through position pattern analysis.
        Real hands have natural movement patterns, false positives often have unnatural patterns."""
        # If we don't have enough history, not suspicious
        if len(self.position_history) < 5:
            return False
            
        # Check last 5 positions
        recent_positions = list(self.position_history)[-5:]
        
        # Calculate movement variance
        x_vals = [p[0] for p in recent_positions]
        y_vals = [p[1] for p in recent_positions]
        
        # If all positions are identical, that's definitely suspicious
        # Real hands always have at least some pixel-level movement
        if len(set(x_vals)) == 1 and len(set(y_vals)) == 1:
            self.still_counter += 1
            print(f"[DEBUG] Suspicious frozen position detected - still_counter: {self.still_counter}")
            return self.still_counter > 2  # After 2 frames of complete stillness, flag as suspicious (reduced from 3)
        
        # Calculate maximum movement in window
        max_x_diff = max(x_vals) - min(x_vals)
        max_y_diff = max(y_vals) - min(y_vals)
        max_movement = max(max_x_diff, max_y_diff)
        
        # Extremely minimal movement over several frames is suspicious
        # Decreased threshold to catch more false positives
        if max_movement < 1.0:  # Less than 1 pixel movement across 5 frames (increased from 0.5)
            self.still_counter += 1
            print(f"[DEBUG] Nearly frozen position detected - max_movement: {max_movement:.2f}px, still_counter: {self.still_counter}")
            return self.still_counter > 3  # Reduced from 5 to 3 frames to flag suspicious patterns earlier
            
        # Check for perfectly regular movement patterns (unnatural)
        # Real hands have irregular micro-movements
        x_diffs = [x_vals[i] - x_vals[i-1] for i in range(1, len(x_vals))]
        y_diffs = [y_vals[i] - y_vals[i-1] for i in range(1, len(y_vals))]
        
        # Calculate regularity - if the differences between points are too consistent, it's suspicious
        if len(x_diffs) >= 3 and len(y_diffs) >= 3:
            x_regularity = max(x_diffs) - min(x_diffs) if max(x_diffs) > 0 else 0
            y_regularity = max(y_diffs) - min(y_diffs) if max(y_diffs) > 0 else 0
            
            # Extremely regular movement is suspicious (common in false positives)
            if x_regularity < 0.2 and y_regularity < 0.2 and max_movement > 0:
                print(f"[DEBUG] Suspiciously regular movement detected: x_reg={x_regularity:.2f}, y_reg={y_regularity:.2f}")
                self.still_counter += 1
                return self.still_counter > 3
        
        # Reset counter if there's natural movement
        self.still_counter = 0
        return False
        
    def reset(self):
        """Reset validator state."""
        self.position_history.clear()
        self.time_history.clear()
        self.still_counter = 0