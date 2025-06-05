"""
Position smoothing utilities for reducing jitter in hand tracking.
"""

import numpy as np
from collections import deque

class CoordinateSmoother:
    """Implements various smoothing algorithms for position data."""
    
    def __init__(self, smoothing_factor=0.4, history_size=5):
        """Initialize coordinate smoother with smoothing parameters."""
        self.smoothing_factor = smoothing_factor
        self.history_size = history_size
        self.position_history = deque(maxlen=history_size)
        self.smoothed_position = None
        
    def exponential_smooth(self, position):
        """Apply exponential smoothing to position data."""
        if position is None:
            return self.smoothed_position
            
        # First detection - initialize smoothed position
        if self.smoothed_position is None:
            self.smoothed_position = position
            self.position_history.append(position)
            return position
            
        # Apply exponential smoothing
        alpha = self.smoothing_factor
        self.smoothed_position = (
            int((1 - alpha) * self.smoothed_position[0] + alpha * position[0]),
            int((1 - alpha) * self.smoothed_position[1] + alpha * position[1])
        )
        
        # Store position in history
        self.position_history.append(position)
        
        return self.smoothed_position
        
    def average_smooth(self, position):
        """Apply simple moving average smoothing."""
        if position is None:
            return self.smoothed_position
            
        # Add position to history
        self.position_history.append(position)
        
        # Not enough points for smoothing
        if len(self.position_history) < 2:
            return position
            
        # Calculate average position
        x_avg = int(sum(pos[0] for pos in self.position_history) / len(self.position_history))
        y_avg = int(sum(pos[1] for pos in self.position_history) / len(self.position_history))
        
        self.smoothed_position = (x_avg, y_avg)
        return self.smoothed_position
        
    def reset(self):
        """Reset smoother state."""
        self.position_history.clear()
        self.smoothed_position = None
