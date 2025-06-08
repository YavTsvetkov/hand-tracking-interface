"""
Drawing functions for visualizing hand landmarks and tracking data.
"""

import cv2
import numpy as np


class Renderer:
    """Handles rendering of visualizations on frames."""
    
    def __init__(self):
        """Initialize renderer."""
        # Define colors for different tracking states
        self.colors = {
            'detected': (0, 255, 0),      # Green for good detection
            'uncertain': (0, 255, 255),    # Yellow for uncertain detection
            'lost': (0, 0, 255),          # Red for lost tracking
            'false_positive': (255, 0, 255)  # Magenta for suspected false positives
        }
        
    def draw_landmarks(self, frame, landmarks, tracking_quality=1.0):
        """Draw hand landmarks on frame - optimized for palm detection visualization."""
        if landmarks is None or len(landmarks) == 0:
            return
            
        # Always use green color for maximum visibility regardless of tracking quality
        # This helps better see the raw positions without quality-based color changes
        color = self.colors['detected']  # Always use green for detected landmarks
            
        # For palm detection, emphasize palm structure
        # Draw points with different sizes based on importance
        for i, point in enumerate(landmarks):
            # Palm landmarks (larger circles)
            if i in [0, 5, 9, 13, 17]:  # Wrist and finger bases (palm outline)
                radius = 10  # Increased size for better visibility
                thickness = -1  # Filled
            elif i in [4, 8, 12, 16, 20]:  # Fingertips
                radius = 8  # Increased size for fingertips
                thickness = -1
            else:
                radius = 4  # Slightly larger for all points for better visibility
                thickness = -1  # Fill all points for better visibility
                
            cv2.circle(frame, point, radius, color, thickness)
            
        # Draw connections - emphasize palm structure for palm detection
        # Palm outline (more prominent for palm detection)
        palm_outline = [0, 5, 9, 13, 17, 0]  # Wrist to finger bases and back
        self._connect_points(frame, landmarks, palm_outline, color, thickness=3)
        
        # Finger connections (thinner lines)
        # Thumb
        self._connect_points(frame, landmarks, [0, 1, 2, 3, 4], color, thickness=2)
        # Index finger
        self._connect_points(frame, landmarks, [5, 6, 7, 8], color, thickness=2)
        # Middle finger
        self._connect_points(frame, landmarks, [9, 10, 11, 12], color, thickness=2)
        # Ring finger
        self._connect_points(frame, landmarks, [13, 14, 15, 16], color, thickness=2)
        # Pinky
        self._connect_points(frame, landmarks, [17, 18, 19, 20], color, thickness=2)
        
    def draw_wrist_position(self, frame, position, tracking_quality=1.0, is_valid=True):
        """Draw wrist position marker."""
        if position is None:
            return
            
        # Determine color based on tracking quality and validity
        if not is_valid:
            color = self.colors['false_positive']
        elif tracking_quality > 0.7:
            color = self.colors['detected']
        else:
            color = self.colors['uncertain']
            
        # Draw wrist position
        cv2.circle(frame, position, 8, color, -1)
        cv2.circle(frame, position, 10, color, 2)
        
    def _connect_points(self, frame, points, indices, color, thickness=2):
        """Connect points in order of indices."""
        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], color, thickness)
            
    def draw_status_overlay(self, frame, tracking_quality=0.0, wrist_position=None,
                          is_hand_present=False, is_valid=True, fps=0):
        """Draw status overlay with tracking information."""
        # Status background
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (210, 110), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (210, 110), (255, 255, 255), 1)
        
        # Tracking status
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
            
        cv2.putText(frame, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.6, color, 2)
        
        # Show position if available
        if wrist_position:
            pos_text = f"Position: {wrist_position[0]}, {wrist_position[1]}"
            cv2.putText(frame, pos_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 1)
        
        # Confidence bar
        bar_width = 180
        filled_width = int(bar_width * tracking_quality)
        cv2.rectangle(frame, (20, 75), (20 + bar_width, 85), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, 75), (20 + filled_width, 85), color, -1)
        cv2.putText(frame, f"Quality: {tracking_quality:.2f}", (20, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 90, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
