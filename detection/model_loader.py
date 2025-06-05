"""
TensorFlow Lite model loading and management.
"""

import os
import numpy as np

# Try to import TFLite runtime first, fall back to TensorFlow if not available
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class ModelLoader:
    """TensorFlow Lite model loader for hand landmark detection."""
    
    def __init__(self, model_path='hand_landmark_lite.tflite'):
        """Initialize model loader with model path."""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """Load the TensorFlow Lite model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get input shape
            self.input_shape = self.input_details[0]['shape'][1:3]  # Typically [1, height, width, channels]
            
            print(f"[INFO] Model loaded: {self.model_path}")
            print(f"[INFO] Input shape: {self.input_shape}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
            
    def get_input_shape(self):
        """Get the required input shape for the model."""
        if self.input_details is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.input_shape
