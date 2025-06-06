#!/usr/bin/env python3
"""
Simple debug script to test solid background detection.
"""

import cv2
import numpy as np

# Import modules
from detection import ModelLoader, InferenceEngine, LandmarkExtractor
from camera import FrameProcessor


def test_solid_background():
    """Test detection with solid color backgrounds."""
    print("[DEBUG] Testing solid background detection...")
    
    # Initialize detection components
    model_loader = ModelLoader(model_path="/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
        
    inference_engine = InferenceEngine(model_loader)
    landmark_extractor = LandmarkExtractor(confidence_threshold=0.2)
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    # Test different backgrounds
    test_cases = [
        ((0, 0, 0), "BLACK"),
        ((255, 255, 255), "WHITE"), 
        ((128, 128, 128), "GRAY"),
        ((255, 0, 0), "RED"),
    ]
    
    for color, name in test_cases:
        print(f"\n[TEST] Testing {name} background...")
        
        # Create solid color image
        width, height = 640, 480
        test_frame = np.full((height, width, 3), color, dtype=np.uint8)
        
        # Process frame
        processed_frame = frame_processor.preprocess(test_frame)
        
        # Prepare input for model
        input_tensor = inference_engine.prepare_input(processed_frame)
        
        # Run inference
        landmarks, handedness, hand_scores, landmark_scores = inference_engine.run_inference(input_tensor)
        
        print(f"  Raw hand_scores: {hand_scores}")
        print(f"  Raw handedness: {handedness}")
        print(f"  Landmark_scores shape: {landmark_scores.shape if landmark_scores is not None else None}")
        
        # Extract landmarks
        pixels, wrist_position, confidence = landmark_extractor.extract_landmarks(
            landmarks, handedness, hand_scores, landmark_scores, test_frame.shape
        )
        
        print(f"  Detection result: {wrist_position is not None}")
        print(f"  Confidence: {confidence:.3f}")
        
        if wrist_position is not None:
            print(f"  ⚠️  FALSE POSITIVE: Hand detected on {name} background!")
        else:
            print(f"  ✅ Correct: No hand detected on {name} background")
        
        # Save test image
        cv2.imwrite(f"/home/rtsvetkov/hand_tracking/test_{name.lower()}.jpg", test_frame)


if __name__ == "__main__":
    test_solid_background()
