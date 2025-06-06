#!/usr/bin/env python3
"""
Detailed analysis of model outputs to understand confidence scoring.
"""

import os
import sys
import time
import cv2
import numpy as np

# Add project path
sys.path.append('.')

from detection.model_loader import ModelLoader
from camera.libcamera_capture import LibcameraCapture
from camera.frame_processor import FrameProcessor
from config.settings import Settings

def analyze_model_outputs():
    """Analyze model outputs in detail to understand confidence mechanism."""
    print("=== Detailed Model Output Analysis ===\n")
    
    # Load model
    print("1. Loading model...")
    model_loader = ModelLoader(model_path="hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
    
    # Print output details
    print("2. Model output structure:")
    for i, output in enumerate(model_loader.output_details):
        print(f"   Output {i}: shape={output['shape']}, dtype={output['dtype']}, name='{output.get('name', 'Unknown')}'")
    
    # Initialize camera and frame processor
    print("\n3. Initializing camera...")
    camera = LibcameraCapture(320, 240, fps=30)
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    print("\n4. Testing with different scenarios:")
    
    scenarios = [
        ("Empty room (no hand)", "real_frame"),
        ("Solid black frame", "black"),
        ("Solid white frame", "white"),
        ("Random noise", "noise")
    ]
    
    for scenario_name, scenario_type in scenarios:
        print(f"\n--- {scenario_name} ---")
        
        if scenario_type == "real_frame":
            # Capture real frame
            ret, frame = camera.read()
            if not ret:
                print("Failed to capture frame")
                continue
        elif scenario_type == "black":
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        elif scenario_type == "white":
            frame = np.ones((240, 320, 3), dtype=np.uint8) * 255
        elif scenario_type == "noise":
            frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
        
        # Process frame
        processed_frame = frame_processor.preprocess(frame)
        
        # Resize to model input size
        target_height, target_width = model_loader.get_input_shape()
        resized_frame = cv2.resize(processed_frame, (target_width, target_height))
        
        # Convert to RGB and normalize
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        normalized_frame = np.float32(rgb_frame) / 255.0
        input_tensor = np.expand_dims(normalized_frame, axis=0)
        
        # Run inference
        model_loader.interpreter.set_tensor(model_loader.input_details[0]['index'], input_tensor)
        model_loader.interpreter.invoke()
        
        # Analyze all outputs
        for i, output in enumerate(model_loader.output_details):
            output_data = model_loader.interpreter.get_tensor(output['index'])
            print(f"  Output {i}: shape={output_data.shape}")
            
            if i == 0:  # Landmarks (typically 63 values)
                if output_data.size == 63:
                    landmarks = output_data.reshape(-1, 3)
                    print(f"    Landmarks (first 3 points): {landmarks[:3]}")
                    print(f"    X range: [{landmarks[:, 0].min():.3f}, {landmarks[:, 0].max():.3f}]")
                    print(f"    Y range: [{landmarks[:, 1].min():.3f}, {landmarks[:, 1].max():.3f}]")
                    print(f"    Z range: [{landmarks[:, 2].min():.3f}, {landmarks[:, 2].max():.3f}]")
                else:
                    print(f"    Raw values: {output_data.flatten()[:5]}...")
            
            elif i == 1:  # Handedness
                print(f"    Handedness: {output_data.flatten()}")
                
            elif i == 2:  # Hand presence/confidence score
                score = output_data.flatten()[0]
                print(f"    Hand presence score: {score:.6f}")
                
            elif i == 3:  # World landmarks or additional data
                if output_data.size == 63:
                    world_landmarks = output_data.reshape(-1, 3)
                    print(f"    World landmarks (first 3 points): {world_landmarks[:3]}")
                else:
                    print(f"    Additional data: {output_data.flatten()[:5]}...")
    
    # Cleanup
    camera.release()
    print("\n=== Analysis complete ===")

if __name__ == "__main__":
    analyze_model_outputs()
