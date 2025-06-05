"""
Movement analysis utilities for hand tracking.
"""

import numpy as np
import math
from collections import deque

class MovementAnalyzer:
    """Analyzes hand movement patterns for gesture detection and tracking improvements."""
    
    def __init__(self, history_size=20):
        """Initialize movement analyzer with tracking history size."""
        self.position_history = deque(maxlen=history_size)
        self.velocity_history = deque(maxlen=history_size-1)
        self.acceleration_history = deque(maxlen=history_size-2)
        self.jerk_history = deque(maxlen=history_size-3)
        
    def add_position(self, position):
        """Add a new position and update movement derivatives."""
        if position is None:
            return
            
        self.position_history.append(position)
        
        # Calculate velocity (first derivative of position)
        if len(self.position_history) >= 2:
            p1 = self.position_history[-1]
            p2 = self.position_history[-2]
            velocity = (p1[0] - p2[0], p1[1] - p2[1])
            self.velocity_history.append(velocity)
            
        # Calculate acceleration (second derivative of position)
        if len(self.velocity_history) >= 2:
            v1 = self.velocity_history[-1]
            v2 = self.velocity_history[-2]
            acceleration = (v1[0] - v2[0], v1[1] - v2[1])
            self.acceleration_history.append(acceleration)
            
        # Calculate jerk (third derivative of position)
        if len(self.acceleration_history) >= 2:
            a1 = self.acceleration_history[-1]
            a2 = self.acceleration_history[-2]
            jerk = (a1[0] - a2[0], a1[1] - a2[1])
            self.jerk_history.append(jerk)
    
    def get_movement_magnitude(self):
        """Get recent movement magnitude."""
        if not self.velocity_history:
            return 0
            
        # Average velocity magnitude over recent history
        v_magnitudes = [math.sqrt(v[0]*v[0] + v[1]*v[1]) for v in self.velocity_history]
        return np.mean(v_magnitudes)
        
    def get_movement_direction(self):
        """Get dominant movement direction in degrees (0-360)."""
        if not self.velocity_history or len(self.velocity_history) < 3:
            return None
            
        # Take average of recent velocities
        vx = sum(v[0] for v in self.velocity_history) / len(self.velocity_history)
        vy = sum(v[1] for v in self.velocity_history) / len(self.velocity_history)
        
        # Convert to angle in degrees
        angle = math.degrees(math.atan2(vy, vx)) % 360
        return angle
        
    def is_movement_consistent(self, threshold=0.7):
        """Check if movement direction is consistent (not random)."""
        if not self.velocity_history or len(self.velocity_history) < 5:
            return False
            
        # Calculate movement consistency using vector dot products
        vectors = list(self.velocity_history)
        
        # Normalize vectors
        normalized = []
        for v in vectors:
            magnitude = math.sqrt(v[0]*v[0] + v[1]*v[1])
            if magnitude > 0.001:  # Avoid division by zero
                normalized.append((v[0]/magnitude, v[1]/magnitude))
            else:
                normalized.append((0, 0))
                
        # Calculate average dot product between adjacent vectors
        dot_products = []
        for i in range(len(normalized)-1):
            v1 = normalized[i]
            v2 = normalized[i+1]
            dot = v1[0]*v2[0] + v1[1]*v2[1]  # Dot product
            dot_products.append(dot)
            
        # Average dot product near 1 means consistent direction
        avg_dot = sum(dot_products) / len(dot_products) if dot_products else 0
        return avg_dot > threshold
        
    def reset(self):
        """Reset analyzer state."""
        self.position_history.clear()
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.jerk_history.clear()