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
        """Run model inference on prepared input tensor."""
        interpreter = self.model_loader.interpreter
        input_details = self.model_loader.input_details
        output_details = self.model_loader.output_details
        
        # Set input tensor
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        
        # Run inference
        interpreter.invoke()
        
        # Get all 4 outputs
        landmarks = interpreter.get_tensor(output_details[0]['index'])
        handedness = interpreter.get_tensor(output_details[1]['index'])
        hand_score = interpreter.get_tensor(output_details[2]['index'])
        landmark_scores = interpreter.get_tensor(output_details[3]['index'])  # NEW: 4th output
        
        return landmarks, handedness, hand_score, landmark_scores
