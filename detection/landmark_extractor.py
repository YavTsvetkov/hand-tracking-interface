"""
Hand landmark extraction and processing - optimized for hand_landmark_lite.tflite
"""

import numpy as np

class LandmarkExtractor:
    """Extracts and processes hand landmarks - optimized for hand_landmark_lite.tflite model."""
    
    def __init__(self, confidence_threshold=0.5, model_input_shape=(192, 192)):
        """Initialize landmark extractor with confidence threshold and model input dimensions."""
        self.confidence_threshold = confidence_threshold
        # Model input dimensions: (height, width)
        self.model_input_shape = model_input_shape
        
    def extract_landmarks(self, landmarks, handedness, hand_scores, landmark_scores, frame_shape):
        """Extract landmarks and convert them to image coordinates - optimized for current model."""
        # Basic validation
        if landmarks is None or hand_scores is None:
            return None, None, 0.0
            
        # Extract confidence score
        confidence = float(hand_scores[0][0])
        print(f"[DEBUG] Landmark extractor: confidence={confidence:.3f}, threshold={self.confidence_threshold}")
        print(f"[DEBUG] landmarks array shape: {getattr(landmarks, 'shape', None)}")
        
        # Apply confidence threshold (confidence is now a probability 0-1)
        if confidence < self.confidence_threshold:
            print(f"[DEBUG] Confidence too low: {confidence:.3f} < {self.confidence_threshold}")
            return None, None, confidence
        
        # Extract hand landmarks - should be [1, 21, 3] format from inference engine
        try:
            hand_landmarks = landmarks[0]  # Expected shape: [21, 3]
            print(f"[DEBUG] hand_landmarks length: {len(hand_landmarks)}")
            # Verify we have the expected 21 landmarks
            if len(hand_landmarks) != 21:
                print(f"[DEBUG] Unexpected landmark count: {len(hand_landmarks)} != 21")
                return None, None, confidence
        
        except Exception as e:
            print(f"[DEBUG] Exception extracting landmarks: {e}")
            return None, None, confidence
        
        # Get image dimensions
        img_height, img_width = frame_shape[:2]
        
        # Convert landmarks to pixel coordinates
        # Assuming normalized coordinates from model
        pixels = []
        # Normalize raw landmark coords (pixel values relative to model input) to [0,1]
        in_h, in_w = self.model_input_shape
        for i in range(21):
            raw_x, raw_y = hand_landmarks[i][0], hand_landmarks[i][1]
            x_norm = raw_x / in_w
            y_norm = raw_y / in_h
            # Convert to original frame pixel coords
            x = int(x_norm * img_width)
            y = int(y_norm * img_height)
            pixels.append((x, y))
        print(f"[DEBUG] first 5 pixel coordinates: {pixels[:5]}")
        wrist_position = pixels[0]
        print(f"[DEBUG] wrist_position candidate: {wrist_position}")
                  
        # Basic bounds check for wrist
        if not (0 <= wrist_position[0] < img_width and 0 <= wrist_position[1] < img_height):
            print(f"[DEBUG] Position outside frame after pixel conversion: {wrist_position}, frame size: {img_width}x{img_height}")
            return None, None, confidence

        # All checks passed
        return pixels, wrist_position, confidence
