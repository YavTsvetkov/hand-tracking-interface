"""
Status display utilities for the hand tracking application.
"""

import cv2
import numpy as np
import time


class StatusDisplay:
    """Handles display of application status and performance information."""
    
    def __init__(self, width=640, height=480):
        """Initialize status display with frame dimensions."""
        self.width = width
        self.height = height
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.line_height = 20
        self.margin = 10
        
        # Status colors
        self.colors = {
            'good': (0, 255, 0),      # Green
            'warning': (0, 255, 255),  # Yellow
            'error': (0, 0, 255),     # Red
            'neutral': (255, 255, 255)  # White
        }
        
        # Performance metrics
        self.fps_history = []
        self.fps_update_time = time.time()
        
    def draw_status_text(self, frame, text, position, color_key='neutral', scale=0.5, thickness=1):
        """Draw status text with drop shadow for better visibility."""
        color = self.colors.get(color_key, self.colors['neutral'])
        
        # Draw shadow (offset black text for better visibility)
        cv2.putText(frame, text, (position[0] + 1, position[1] + 1), 
                    self.font, scale, (0, 0, 0), thickness + 1)
                    
        # Draw text
        cv2.putText(frame, text, position, self.font, scale, color, thickness)
        
    def create_status_overlay(self, fps=0, hand_present=False, detection_confidence=0, tracking_quality=0,
                            suspiciously_still=False, valid_position=True):
        """Create a separate status overlay image."""
        # Create transparent overlay
        overlay = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Draw semi-transparent background for status area
        status_width = 220
        status_height = 150
        cv2.rectangle(overlay, (self.width - status_width - 10, 10), 
                    (self.width - 10, status_height + 10), (30, 30, 30), -1)
        
        # Title
        title = "Hand Tracking Status"
        self.draw_status_text(overlay, title, (self.width - status_width, 30), 'neutral', 0.6, 2)
        
        # FPS indicator
        y_pos = 30 + self.line_height
        fps_text = f"FPS: {fps:.1f}"
        fps_color = 'good' if fps >= 15 else ('warning' if fps >= 5 else 'error')
        self.draw_status_text(overlay, fps_text, (self.width - status_width, y_pos), fps_color)
        
        # Hand detection status
        y_pos += self.line_height
        if hand_present:
            status_text = "Status: Hand detected"
            status_color = 'good'
        else:
            status_text = "Status: No hand detected"
            status_color = 'neutral'
        self.draw_status_text(overlay, status_text, (self.width - status_width, y_pos), status_color)
        
        # Confidence
        y_pos += self.line_height
        conf_text = f"Confidence: {detection_confidence:.2f}"
        conf_color = 'good' if detection_confidence > 0.7 else ('warning' if detection_confidence > 0.5 else 'error')
        self.draw_status_text(overlay, conf_text, (self.width - status_width, y_pos), conf_color)
        
        # Tracking quality
        y_pos += self.line_height
        quality_text = f"Tracking quality: {tracking_quality:.2f}"
        quality_color = 'good' if tracking_quality > 0.7 else ('warning' if tracking_quality > 0.5 else 'error')
        self.draw_status_text(overlay, quality_text, (self.width - status_width, y_pos), quality_color)
        
        # Warning indicators
        y_pos += self.line_height
        if suspiciously_still:
            self.draw_status_text(overlay, "Warning: Position not changing", 
                                (self.width - status_width, y_pos), 'warning')
            y_pos += self.line_height
            
        if not valid_position:
            self.draw_status_text(overlay, "Warning: Invalid position detected", 
                                (self.width - status_width, y_pos), 'error')
        
        return overlay
    
    def update_fps(self, current_fps):
        """Update FPS history for smoothed display."""
        current_time = time.time()
        
        # Only keep recent FPS values (last 2 seconds)
        self.fps_history.append((current_time, current_fps))
        self.fps_history = [x for x in self.fps_history if current_time - x[0] < 2.0]
        
    def get_average_fps(self):
        """Get smoothed FPS value from history."""
        if not self.fps_history:
            return 0
            
        # Calculate average of recent FPS values
        fps_values = [x[1] for x in self.fps_history]
        return sum(fps_values) / len(fps_values)
        
    def draw_debug_info(self, frame, debug_values):
        """Draw detailed debug information."""
        if not debug_values:
            return
            
        # Create transparent debug overlay
        overlay = np.zeros_like(frame)
        
        # Background for debug area
        cv2.rectangle(overlay, (10, self.height - 110), 
                    (300, self.height - 10), (30, 30, 30), -1)
                    
        # Draw debug values
        y_pos = self.height - 90
        for key, value in debug_values.items():
            text = f"{key}: {value}"
            cv2.putText(overlay, text, (20, y_pos), self.font, 0.5, (255, 255, 255), 1)
            y_pos += 20
            
        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
    def draw_help_text(self, frame):
        """Draw help text on the frame."""
        help_text = [
            "Press 'q' to quit",
            "Press 'd' to toggle debug info",
            "Press 'h' to toggle help"
        ]
        
        # Create semitransparent overlay
        overlay = np.zeros_like(frame)
        cv2.rectangle(overlay, (10, 10), (200, 90), (30, 30, 30), -1)
        
        # Draw help text
        for i, text in enumerate(help_text):
            cv2.putText(overlay, text, (20, 35 + i * 20), self.font, 0.5, (255, 255, 255), 1)
            
        # Apply overlay with transparency
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
