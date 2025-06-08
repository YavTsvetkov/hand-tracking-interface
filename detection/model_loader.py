"""
TensorFlow Lite model loading and management - optimized for palm_detection.tflite
"""

import os
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class ModelLoader:
    """TensorFlow Lite model loader optimized for palm_detection.tflite."""
    
    def __init__(self, model_path='models/palm_detection.tflite'):
        """Initialize model loader with fixed model path."""
        self.model_path = model_path
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        
    def load_model(self):
        """Load the palm_detection.tflite model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        try:
            self.interpreter = Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Palm detection models typically have multiple outputs
            # Expected outputs: detections (boxes), scores, possibly classifications
            print(f"[INFO] Palm detection model has {len(self.output_details)} outputs")
            
            # Get input shape - palm detection models often use 192x192 or 256x256
            self.input_shape = self.input_details[0]['shape'][1:3]  # [height, width]
            
            print(f"[INFO] Model loaded: {self.model_path}")
            print(f"[INFO] Input shape: {self.input_shape}")
            print(f"[INFO] Model type: Palm Detection")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False
            
    def get_input_shape(self):
        """Get the required input shape for the model (typically 192x192 or 256x256 for palm detection)."""
        if self.input_details is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self.input_shape
