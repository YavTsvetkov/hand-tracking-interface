#!/usr/bin/env python3
"""
Debug script for testing hand landmark model.
This will help diagnose issues with hand detection.
"""

import cv2
import numpy as np
import os
import time
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our modules
from config.settings import Settings
from camera.libcamera_capture import LibcameraCapture
from camera.frame_processor import FrameProcessor
from detection.model_loader import ModelLoader

# Try to load TFLite 
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print("Using TensorFlow instead of TFLite Runtime")
    except ImportError:
        print("Error: Neither TFLite Runtime nor TensorFlow available")
        sys.exit(1)

def run_debug_test():
    """Run debug test for hand tracking model."""
    print("=== Hand Detection Debug Test ===")
    
    # 1. Test model loading
    print("\n1. Testing model loading...")
    model_path = "hand_landmark_lite.tflite"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found: {model_path}")
        return False
    
    try:
        model_loader = ModelLoader(model_path=model_path)
        if not model_loader.load_model():
            print("ERROR: Failed to load model")
            return False
        print("✓ Model loaded successfully")
        
        # Get model details
        input_details = model_loader.input_details
        output_details = model_loader.output_details
        print(f"- Input shape: {input_details[0]['shape']}")
        print(f"- Input type: {input_details[0]['dtype']}")
        
        for i, output in enumerate(output_details):
            print(f"- Output {i} shape: {output['shape']}")
            print(f"- Output {i} type: {output['dtype']}")
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    # 2. Initialize camera
    print("\n2. Testing camera initialization...")
    try:
        resolution = "320x240"  # Use lower resolution for testing
        width, height = Settings.get_resolution_as_tuple(resolution)
        camera = LibcameraCapture(width, height, fps=30)
        print(f"✓ Camera initialized at {width}x{height}")
    except Exception as e:
        print(f"ERROR initializing camera: {e}")
        return False
    
    # 3. Test frame capture
    print("\n3. Testing frame capture...")
    try:
        print("- Attempting to capture frame (this may take a moment)...")
        ret, frame = camera.read()
        if not ret or frame is None:
            print("ERROR: Failed to capture frame")
            camera.release()
            return False
        print(f"✓ Frame captured: {frame.shape}")
        
        # Save raw frame for visual inspection
        cv2.imwrite("debug_raw_frame.jpg", frame)
        print("- Saved raw frame to debug_raw_frame.jpg")
    except Exception as e:
        print(f"ERROR capturing frame: {e}")
        camera.release()
        return False
    
    # 4. Test preprocessing
    print("\n4. Testing frame preprocessing...")
    try:
        frame_processor = FrameProcessor(crop_factor=0.8)
        processed_frame = frame_processor.preprocess(frame)
        print(f"✓ Frame preprocessed: {processed_frame.shape}")
        
        # Save processed frame
        cv2.imwrite("debug_processed_frame.jpg", cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
        print("- Saved processed frame to debug_processed_frame.jpg")
    except Exception as e:
        print(f"ERROR preprocessing frame: {e}")
        camera.release()
        return False
    
    # 5. Test model input preparation
    print("\n5. Testing model input preparation...")
    try:
        # Get required input shape
        input_shape = model_loader.get_input_shape()
        print(f"- Required input shape: {input_shape}")
        
        # Resize to match model input
        resized_frame = cv2.resize(processed_frame, (input_shape[1], input_shape[0]))
        print(f"✓ Resized frame: {resized_frame.shape}")
        
        # Normalize
        normalized_frame = np.float32(resized_frame) / 255.0
        print(f"✓ Normalized frame (min={normalized_frame.min():.2f}, max={normalized_frame.max():.2f})")
        
        # Add batch dimension
        input_tensor = np.expand_dims(normalized_frame, axis=0)
        print(f"✓ Input tensor: {input_tensor.shape}")
        
        # Save input tensor visualization
        cv2.imwrite("debug_model_input.jpg", cv2.cvtColor(resized_frame, cv2.COLOR_RGB2BGR))
        print("- Saved input tensor visualization to debug_model_input.jpg")
    except Exception as e:
        print(f"ERROR preparing model input: {e}")
        camera.release()
        return False
    
    # 6. Test model inference
    print("\n6. Testing model inference...")
    try:
        # Set input tensor
        model_loader.interpreter.set_tensor(input_details[0]['index'], input_tensor)
        
        # Run inference
        start_time = time.time()
        model_loader.interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        print(f"✓ Inference completed in {inference_time:.2f}ms")
        
        # Get outputs
        for i, output in enumerate(output_details):
            output_data = model_loader.interpreter.get_tensor(output['index'])
            print(f"- Output {i} shape: {output_data.shape}")
            
            if i == 0:  # Landmarks
                print(f"- Landmarks output: shape={output_data.shape}")
                # Print the first landmark coordinate
                if output_data.size > 0:
                    if output_data.ndim >= 3:
                        first_landmark = output_data[0][0][0]
                        print(f"- First landmark: {first_landmark}")
                    else:
                        print(f"- Output structure is different: {output_data.ndim} dimensions")
            
            if i == 2:  # Hand scores
                print(f"- Hand score: {output_data[0][0]:.4f}")
        
    except Exception as e:
        print(f"ERROR during inference: {e}")
        camera.release()
        return False
    
    # Clean up
    print("\n7. Cleaning up resources...")
    camera.release()
    print("✓ Camera resources released")
    
    print("\n=== Debug test completed ===")
    print("Check the saved debug images to verify camera and model operation")
    return True

if __name__ == "__main__":
    run_debug_test()
