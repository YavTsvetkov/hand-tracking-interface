"""
Status display and overlays for tracking application.
"""

import cv2
import numpy as np
import time


class StatusDisplay:
    """Handles display of tracking status and performance metrics."""
    
    def __init__(self, width=640, height=480):
        """Initialize status display with frame dimensions."""
        self.width = width
        self.height = height
        
        # Colors for different states
        self.colors = {
            'detected': (0, 255, 0),       # Green for detected
            'uncertain': (0, 255, 255),    # Yellow for uncertain detection
            'lost': (0, 0, 255),           # Red for lost tracking
            'false_positive': (255, 0, 255)  # Magenta for false positives
        }
        
        # Fonts
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
    def draw_status_box(self, frame, is_hand_present, tracking_quality, 
                       position=None, is_valid=True, fps=0):
        """Draw main status box with tracking information."""
        # Status box background
        cv2.rectangle(frame, (10, 10), (230, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (230, 100), (255, 255, 255), 1)
        
        # Determine status text and color
        if is_hand_present:
            if is_valid:
                if tracking_quality > 0.7:
                    status = "DETECTED ✓"
                    color = self.colors['detected']
                else:
                    status = "UNCERTAIN"
                    color = self.colors['uncertain']
            else:
                status = "FALSE POSITIVE!"
                color = self.colors['false_positive']
        else:
            status = "NOT DETECTED"
            color = self.colors['lost']
            
        # Draw status text
        cv2.putText(frame, status, (20, 35), self.font, 0.7, color, 2)
        
        # Draw position if available
        if position is not None:
            pos_text = f"Position: {position[0]}, {position[1]}"
            cv2.putText(frame, pos_text, (20, 60), self.font, 0.5, (255, 255, 255), 1)
            
        # Draw tracking quality bar
        bar_width = 180
        filled_width = int(bar_width * tracking_quality)
        cv2.rectangle(frame, (20, 75), (20 + bar_width, 85), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 75), (20 + filled_width, 85), color, -1)
        
        # Draw FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (self.width - 100, 30), self.font, 
                    0.5, (255, 255, 255), 1)
                    
    def draw_notification(self, frame, message, duration=2.0, color=None):
        """Draw temporary notification message."""
        if color is None:
            color = (255, 255, 255)  # Default to white
            
        # Draw notification background
        text_size = cv2.getTextSize(message, self.font, 0.7, 2)[0]
        box_width = text_size[0] + 40
        
        # Center the notification
        box_start_x = (self.width - box_width) // 2
        
        cv2.rectangle(frame, (box_start_x, 50), (box_start_x + box_width, 90), 
                     (0, 0, 0), -1)
        cv2.rectangle(frame, (box_start_x, 50), (box_start_x + box_width, 90), 
                     color, 2)
                     
        # Draw text
        text_x = box_start_x + 20
        cv2.putText(frame, message, (text_x, 75), self.font, 0.7, color, 2)
        
    def draw_warning(self, frame, message):
        """Draw warning message with warning colors."""
        self.draw_notification(frame, message, color=self.colors['uncertain'])
        
    def draw_error(self, frame, message):
        """Draw error message with error colors."""
        self.draw_notification(frame, message, color=self.colors['lost'])