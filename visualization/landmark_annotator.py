"""
Landmark annotation utilities for hand tracking visualization.
"""

import cv2
import numpy as np


class LandmarkAnnotator:
    """Specialized utility for visualizing hand landmarks."""
    
    def __init__(self):
        """Initialize landmark annotator."""
        # Define landmark connections for drawing
        self.connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm
            (5, 9), (9, 13), (13, 17)
        ]
        
        # Define landmark names for labels
        self.landmark_names = {
            0: "WRIST",
            1: "THUMB_CMC", 2: "THUMB_MCP", 3: "THUMB_IP", 4: "THUMB_TIP",
            5: "INDEX_MCP", 6: "INDEX_PIP", 7: "INDEX_DIP", 8: "INDEX_TIP",
            9: "MIDDLE_MCP", 10: "MIDDLE_PIP", 11: "MIDDLE_DIP", 12: "MIDDLE_TIP",
            13: "RING_MCP", 14: "RING_PIP", 15: "RING_DIP", 16: "RING_TIP",
            17: "PINKY_MCP", 18: "PINKY_PIP", 19: "PINKY_DIP", 20: "PINKY_TIP"
        }
        
    def draw_landmarks(self, frame, landmarks, color=(0, 255, 0), thickness=2):
        """Draw landmarks and connections between them."""
        if landmarks is None or len(landmarks) == 0:
            return
            
        # Draw connections
        for connection in self.connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                cv2.line(frame, landmarks[start_idx], landmarks[end_idx], color, thickness)
                
        # Draw landmark points
        for i, point in enumerate(landmarks):
            # Larger circle for wrist, smaller for other points
            radius = 5 if i == 0 else 3
            cv2.circle(frame, point, radius, color, -1)
            
    def draw_landmark_labels(self, frame, landmarks, color=(255, 255, 255), font_scale=0.3):
        """Draw landmark labels for educational purposes."""
        if landmarks is None:
            return
            
        for i, point in enumerate(landmarks):
            if i in self.landmark_names:
                label = self.landmark_names[i]
                cv2.putText(frame, label, (point[0] + 5, point[1] - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
                            
    def highlight_finger(self, frame, landmarks, finger_indices, color=(0, 255, 255), thickness=3):
        """Highlight a specific finger in the landmarks."""
        if landmarks is None or len(landmarks) == 0:
            return
            
        # Draw connections for the specific finger
        for i in range(len(finger_indices) - 1):
            start_idx = finger_indices[i]
            end_idx = finger_indices[i + 1]
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                cv2.line(frame, landmarks[start_idx], landmarks[end_idx], color, thickness)
                
        # Draw points for the specific finger
        for idx in finger_indices:
            if idx < len(landmarks):
                cv2.circle(frame, landmarks[idx], 5, color, -1)
                
    def draw_hand_orientation(self, frame, landmarks, color=(0, 220, 255), thickness=2):
        """Draw hand orientation indicators (palm normal, etc.)."""
        if landmarks is None or len(landmarks) < 21:
            return
            
        # Get wrist and palm center
        wrist = landmarks[0]
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        middle_mcp = landmarks[9]
        
        # Calculate palm center
        palm_center = (
            (index_mcp[0] + middle_mcp[0] + pinky_mcp[0]) // 3,
            (index_mcp[1] + middle_mcp[1] + pinky_mcp[1]) // 3
        )
        
        # Draw palm center
        cv2.circle(frame, palm_center, 5, color, -1)
        
        # Draw palm orientation line
        cv2.line(frame, wrist, palm_center, color, thickness + 1)
        
        # Calculate and draw palm normal vector
        # (simplified approximation using cross product of palm vectors)
        palm_vector = (palm_center[0] - wrist[0], palm_center[1] - wrist[1])
        normal_length = int(np.sqrt(palm_vector[0]**2 + palm_vector[1]**2) * 0.7)
        normal_vector = (-palm_vector[1], palm_vector[0])  # 90 degree rotation
        
        # Normalize and scale
        magnitude = np.sqrt(normal_vector[0]**2 + normal_vector[1]**2)
        if magnitude > 0:
            normal_vector = (
                int(normal_vector[0] / magnitude * normal_length),
                int(normal_vector[1] / magnitude * normal_length)
            )
            
            normal_end = (
                palm_center[0] + normal_vector[0], 
                palm_center[1] + normal_vector[1]
            )
            
            cv2.arrowedLine(frame, palm_center, normal_end, color, thickness)