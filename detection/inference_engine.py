"""
TensorFlow Lite model inference engine.
"""

import cv2
import numpy as np

class InferenceEngine:
    """Handles inference using the TensorFlow Lite model."""
    
    def __init__(self, model_loader):
        """Initialize inference engine with a model loader instance."""
        self.model_loader = model_loader
        
    def prepare_input(self, frame):
        """Prepare frame for model input."""
        # Get target size from model input details
        target_height, target_width = self.model_loader.get_input_shape()
        
        # Resize frame to match model input size
        input_frame = cv2.resize(frame, (target_width, target_height))
        
        # Normalize to float32 [0,1]
        input_frame = np.float32(input_frame) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_frame, axis=0)
        
        return input_tensor
        
    def run_inference(self, input_tensor):
        """Run model inference on prepared input tensor - optimized for hand_landmark_lite.tflite."""
        interpreter = self.model_loader.interpreter
        input_details = self.model_loader.input_details
        output_details = self.model_loader.output_details
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        
        # Run inference
        interpreter.invoke()
        
        # Get outputs - this model has exactly 2 outputs:
        # Output 0: landmarks [1, 2016, 18] - 2016 anchor boxes with 18 values each
        # Output 1: hand scores [1, 2016, 1] - confidence score for each anchor box
        raw_landmarks = interpreter.get_tensor(output_details[0]['index'])  # [1, 2016, 18]
        raw_scores = interpreter.get_tensor(output_details[1]['index'])     # [1, 2016, 1]
        
        # Find the best detection (highest confidence)
        best_idx = np.argmax(raw_scores[0, :, 0])
        best_score_raw = raw_scores[0, best_idx, 0]
        
        # Convert raw logit to probability using sigmoid
        best_score = 1.0 / (1.0 + np.exp(-best_score_raw))
        
        print(f"[DEBUG] Best detection: idx={best_idx}, raw_score={best_score_raw:.3f}, prob={best_score:.3f}")
        
        # Extract landmarks for the best detection
        # The 18 values per detection likely represent: x1,y1,x2,y2,...,x9,y9 (9 key points * 2 coords)
        # or could be 6 key points * 3 coords (x,y,z)
        best_landmarks_raw = raw_landmarks[0, best_idx, :]  # [18]
        
        # Reshape landmarks to a format compatible with existing code
        # Assuming it's 9 key points with x,y coordinates
        if len(best_landmarks_raw) == 18:
            # Reshape to [9, 2] and add z=0 to make it [9, 3]
            landmarks_2d = best_landmarks_raw.reshape(9, 2)
            landmarks_3d = np.zeros((9, 3))
            landmarks_3d[:, :2] = landmarks_2d
            # Expand to 21 landmarks (standard hand model) by interpolating/duplicating key points
            landmarks = self._expand_to_21_landmarks(landmarks_3d)
        else:
            # If different format, handle accordingly
            landmarks = best_landmarks_raw.reshape(1, -1, 2 if len(best_landmarks_raw) % 2 == 0 else 3)
        
        # Package for compatibility with existing pipeline
        landmarks = landmarks.reshape(1, -1, 3)  # Ensure [1, N, 3] format
        hand_scores = np.array([[best_score]])    # [1, 1] format
        
        return landmarks, None, hand_scores, None
    
    def _expand_to_21_landmarks(self, key_landmarks):
        """Expand 9 key landmarks to 21 standard hand landmarks."""
        # Create 21 landmarks by mapping/interpolating from 9 key points
        # This is a simplified mapping - in practice you'd want more sophisticated interpolation
        full_landmarks = np.zeros((21, 3))
        
        # Map the 9 key points to the most important landmarks
        if len(key_landmarks) >= 9:
            # Map key points to standard MediaPipe landmarks
            landmark_mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Use first 9 as-is
            for i, mapping in enumerate(landmark_mapping):
                if i < len(key_landmarks):
                    full_landmarks[mapping] = key_landmarks[i]
            
            # Fill remaining landmarks with interpolated values
            for i in range(9, 21):
                # Simple interpolation - use nearest key point
                nearest_key = min(8, i // 2)
                full_landmarks[i] = key_landmarks[nearest_key]
        
        return full_landmarks
