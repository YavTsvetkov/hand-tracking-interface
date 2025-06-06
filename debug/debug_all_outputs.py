#!/usr/bin/env python3
"""
Debug script to analyze ALL model outputs including the 4th one we're ignoring.
"""

import numpy as np
import cv2
from detection.model_loader import ModelLoader

def create_test_image(color, size=(224, 224)):
    """Create a solid color test image."""
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)

def analyze_all_outputs():
    """Analyze all 4 model outputs."""
    print("[DEBUG] Analyzing ALL model outputs...")
    
    # Load model
    model_path = "/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite"
    model_loader = ModelLoader(model_path)
    model_loader.load_model()
    
    interpreter = model_loader.interpreter
    input_details = model_loader.input_details
    output_details = model_loader.output_details
    
    print(f"[INFO] Model has {len(output_details)} outputs:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: shape={detail['shape']}, dtype={detail['dtype']}, name={detail.get('name', 'unknown')}")
    
    # Test gray background (known false positive)
    gray_image = create_test_image([128, 128, 128])
    
    # Prepare input
    input_frame = np.float32(gray_image) / 255.0
    input_tensor = np.expand_dims(input_frame, axis=0)
    
    # Set input and run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    # Get ALL outputs
    print(f"\n[GRAY BACKGROUND ANALYSIS]")
    all_outputs = []
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])
        all_outputs.append(output)
        print(f"  Output {i} ({detail.get('name', 'unknown')}):")
        print(f"    Shape: {output.shape}")
        print(f"    Values: {output}")
        if output.size <= 10:  # Only print small outputs fully
            print(f"    Full data: {output.flatten()}")
        else:
            print(f"    Sample (first 10): {output.flatten()[:10]}")
            print(f"    Min/Max: {output.min():.6f} / {output.max():.6f}")
        print()
    
    # Test with black background (should be negative)
    black_image = create_test_image([0, 0, 0])
    input_frame = np.float32(black_image) / 255.0
    input_tensor = np.expand_dims(input_frame, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    
    print(f"\n[BLACK BACKGROUND COMPARISON]")
    for i, detail in enumerate(output_details):
        output = interpreter.get_tensor(detail['index'])
        print(f"  Output {i}: {output.flatten()[:5] if output.size > 5 else output.flatten()}")
    
    return all_outputs

if __name__ == "__main__":
    analyze_all_outputs()
