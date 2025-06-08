"""
Landmark extractor for converting model outputs to pixel coordinates.
This is a reconstruction based on usage patterns in the codebase.
"""

import numpy as np

class LandmarkExtractor:
    """Extracts and converts landmarks from model outputs to pixel coordinates."""
    
    def __init__(self, confidence_threshold=0.5):
        """Initialize the landmark extractor.
        
        Args:
            confidence_threshold: Minimum confidence for valid detections
        """
        self.confidence_threshold = confidence_threshold
        
    def extract_landmarks(self, landmarks, handedness, hand_scores, landmark_scores, frame_shape):
        """Extract landmarks and convert to pixel coordinates.
        
        Args:
            landmarks: Raw landmark data from model [batch, landmarks, 3]
            handedness: Hand classification (left/right)
            hand_scores: Confidence scores for hand detection
            landmark_scores: Confidence scores for individual landmarks
            frame_shape: Shape of the input frame (height, width, channels)
            
        Returns:
            tuple: (pixels, wrist_position, confidence)
                - pixels: All landmarks in pixel coordinates
                - wrist_position: Wrist center in pixel coordinates (x, y)
                - confidence: Detection confidence score
        """
        
        if landmarks is None or hand_scores is None:
            return None, None, 0.0
            
        # Get frame dimensions
        frame_height, frame_width = frame_shape[:2]
        
        # Get the best detection
        if len(hand_scores.shape) > 1:
            best_idx = np.argmax(hand_scores[0])
            confidence = float(hand_scores[0][best_idx])
        else:
            best_idx = 0
            confidence = float(hand_scores[0]) if len(hand_scores) > 0 else 0.0
         
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return None, None, confidence
         
        # Extract landmarks for best detection
        if len(landmarks.shape) == 3:
            landmark_set = landmarks[0]  # Remove batch dimension
        else:
            landmark_set = landmarks

        if len(landmark_set) == 0:
            return None, None, confidence
         
        # Convert landmarks to pixel coordinates
        pixels = []
        for i, landmark in enumerate(landmark_set):
            if len(landmark) >= 2:
                x, y = landmark[0], landmark[1]
                
                # The coordinates might already be in pixel space or normalized
                # Check if they appear to be normalized (0-1 range)
                if x <= 1.0 and y <= 1.0 and x >= 0.0 and y >= 0.0:
                    # Convert from normalized to pixel coordinates
                    pixel_x = int(x * frame_width)
                    pixel_y = int(y * frame_height)
                else:
                    # Assume already in pixel coordinates
                    pixel_x = int(x)
                    pixel_y = int(y)
                     
                pixels.append((pixel_x, pixel_y))
            else:
                pixels.append((0, 0))
         
        # Extract wrist position (typically landmark 0)
        wrist_position = None
        if len(pixels) > 0:
            wrist_position = pixels[0]  # Wrist is typically the first landmark
         
        return pixels, wrist_position, confidence
