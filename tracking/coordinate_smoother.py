"""
Position smoothing utilities for reducing jitter in hand tracking.
"""

import numpy as np
from collections import deque

class CoordinateSmoother:
    """Implements ultra-minimal smoothing for maximum raw position accuracy."""
    
    def __init__(self, smoothing_factor=0.0, history_size=2):
        """Initialize coordinate smoother with absolute minimal history."""
        # Completely ignore smoothing factor - we want raw accuracy
        self.history_size = 2  # Only use 2 frames for absolute minimal filtering
        self.position_history = deque(maxlen=2)  # Fixed size for just 2 frames
        self.smoothed_position = None
        
    def exponential_smooth(self, position):
        """Ultra-minimalist filter - either use raw position or very slight averaging.
        Optimized for maximum raw position accuracy with bare minimum jitter filtering."""
        if position is None:
            return self.smoothed_position
            
        # Store raw position in history
        self.position_history.append(position)
        
        # First detection - always use raw position with no modification
        if len(self.position_history) < 2:
            self.smoothed_position = position
            return position
            
        # Ultra-minimal smoothing - weighted heavily toward raw position (90% current, 10% previous)
        # This preserves almost all raw accuracy while preventing only the most extreme jitter
        recent_positions = list(self.position_history)
        
        # Apply 90/10 weighting toward current position
        # This is barely filtering at all - almost raw data
        current_pos = recent_positions[-1]  # Most recent position
        prev_pos = recent_positions[-2]    # Previous position
        
        # Calculate weighted average with 90% weight on current position
        avg_x = current_pos[0] * 0.9 + prev_pos[0] * 0.1
        avg_y = current_pos[1] * 0.9 + prev_pos[1] * 0.1
        
        # Round to whole pixels at the end to match camera resolution precision
        self.smoothed_position = (round(avg_x), round(avg_y))
        
        return self.smoothed_position
        
    def average_smooth(self, position):
        """Apply ultra-minimal smoothing optimized for raw position accuracy."""
        if position is None:
            return self.smoothed_position
            
        # Add position to history
        self.position_history.append(position)
        
        # First detection - always use raw position with no modification
        if len(self.position_history) < 2:
            return position
            
        # Use same minimal smoothing approach as exponential_smooth
        # Heavily weighted toward current raw position (90%)
        recent_positions = list(self.position_history)
        current_pos = recent_positions[-1]  # Most recent position
        prev_pos = recent_positions[-2]    # Previous position
        
        # Calculate weighted average with 90% weight on current position
        x_avg = current_pos[0] * 0.9 + prev_pos[0] * 0.1
        y_avg = current_pos[1] * 0.9 + prev_pos[1] * 0.1
        
        self.smoothed_position = (round(x_avg), round(y_avg))
        return self.smoothed_position
        
    def reset(self):
        """Reset smoother state."""
        self.position_history.clear()
        self.smoothed_position = None
