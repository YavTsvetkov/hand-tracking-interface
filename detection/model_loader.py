"""
TensorFlow Lite model loading and management - optimized for hand_landmark_lite.tflite
"""

import os
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class ModelLoader:
    """TensorFlow Lite model loader optimized for hand_landmark_lite.tflite."""
    
    def __init__(self, model_path='models/hand_landmark_lite.tflite'):
        """Initialize model loader with fixed model path."""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """Load the hand_landmark_lite.tflite model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Verify this is the expected model format
            expected_outputs = 2
            if len(self.output_details) != expected_outputs:
                print(f"[WARNING] Expected {expected_outputs} outputs, got {len(self.output_details)}")
            
            # Get input shape - should be [1, 192, 192, 3] for hand_landmark_lite
            self.input_shape = self.input_details[0]['shape'][1:3]  # [192, 192]
            
            print(f"[INFO] Model loaded: {self.model_path}")
            print(f"[INFO] Input shape: {self.input_shape}")
            print(f"[INFO] Expected format: hand_landmark_lite.tflite with 2016 anchor boxes")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
            
    def get_input_shape(self):
        """Get the required input shape for the model (192x192 for hand_landmark_lite)."""
        if self.input_details is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.input_shape
