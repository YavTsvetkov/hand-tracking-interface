"""
Hand landmark extraction and processing.
"""

import numpy as np

class LandmarkExtractor:
    """Extracts and processes hand landmarks from model inference results."""
    
    def __init__(self, confidence_threshold=0.6):
        """Initialize landmark extractor with confidence threshold."""
        self.confidence_threshold = confidence_threshold
        
    def extract_landmarks(self, landmarks, handedness, hand_scores, frame_shape):
        """Extract landmarks and convert them to image coordinates."""
        if landmarks is None or len(landmarks) == 0 or hand_scores is None:
            return None, None, 0.0
            
        # Get confidence score
        confidence = float(hand_scores[0][0])
        
        # Skip low confidence detections
        if confidence < self.confidence_threshold:
            return None, None, confidence
            
        # Extract hand landmarks (21 points)
        # The landmarks array shape could vary depending on the model
        # Typically shape is [1, 1, 21, 3] for one hand with 21 points and x,y,z coordinates
        try:
            hand_landmarks = landmarks[0]
            
            # Handle different possible landmark formats
            if hand_landmarks.ndim == 3:  # [1, 21, 3] or similar
                hand_landmarks = hand_landmarks[0]
            elif hand_landmarks.ndim == 1:  # Flattened array
                # Reshape to [21, 3] if it's a flattened array of 63 values (21 points × 3 coordinates)
                if len(hand_landmarks) == 63:
                    hand_landmarks = hand_landmarks.reshape(21, 3)
                else:
                    # If we can't determine the format, return no detection
                    return None, None, confidence
        except Exception as e:
            return None, None, confidence
        
        # Get image dimensions
        img_height, img_width = frame_shape[:2]
        
        # Check if coordinates are already in pixel space or normalized (0-1)
        first_x, first_y = hand_landmarks[0][0], hand_landmarks[0][1]
        is_normalized = first_x <= 1.0 and first_y <= 1.0
        
        # Convert coordinates appropriately
        if is_normalized:
            # Convert normalized coordinates to pixel coordinates
            print(f"[DEBUG] Converting normalized coordinates to pixels")
            pixels = [(int(hand_landmarks[i][0] * img_width), int(hand_landmarks[i][1] * img_height)) 
                      for i in range(len(hand_landmarks))]
        else:
            # Already in pixel space, just convert to int
            print(f"[DEBUG] Using pixel coordinates directly")
            pixels = [(int(hand_landmarks[i][0]), int(hand_landmarks[i][1])) 
                      for i in range(len(hand_landmarks))]
                  
        # Check if points are within image bounds (use only wrist point for check)
        wrist = pixels[0]
        if not (0 <= wrist[0] < img_width and 0 <= wrist[1] < img_height):
            return None, None, confidence
            
        # Extract wrist position (landmark 0 is the wrist)
        wrist_position = wrist
        
        return pixels, wrist_position, confidence
        
    def calculate_critical_confidence(self, landmarks, critical_indices=[0, 5, 9, 13, 17]):
        """Calculate confidence focusing on critical landmarks (wrist and finger bases)."""
        if landmarks is None or len(landmarks) == 0:
            return 0.0
            
        # Get critical points (wrist and finger bases)
        critical_points = [landmarks[i] for i in critical_indices]
        
        # Calculate average z-value (MediaPipe uses z for confidence)
        critical_confidence = np.mean([point[2] for point in critical_points])
        
        return critical_confidence
