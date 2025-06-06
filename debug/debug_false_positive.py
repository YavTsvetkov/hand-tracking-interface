#!/usr/bin/env python3
"""
Detailed analysis of why false positives occur on solid backgrounds.
"""

import cv2
import numpy as np

# Import modules
from detection import ModelLoader, InferenceEngine, LandmarkExtractor
from detection.improved_confidence_analyzer import ConfidenceAnalyzer
from camera import FrameProcessor


def analyze_false_positive():
    """Analyze why false positives occur on solid backgrounds."""
    print("[DEBUG] Analyzing false positive detection...")
    
    # Initialize detection components
    model_loader = ModelLoader(model_path="/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
        
    inference_engine = InferenceEngine(model_loader)
    analyzer = ConfidenceAnalyzer()
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    # Test with gray background (known false positive)
    print(f"\n[ANALYSIS] Testing GRAY background (known false positive)...")
    
    # Create gray image
    width, height = 640, 480
    test_frame = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    
    # Process frame
    processed_frame = frame_processor.preprocess(test_frame)
    
    # Prepare input for model
    input_tensor = inference_engine.prepare_input(processed_frame)
    
    # Run inference
    landmarks, handedness, hand_scores = inference_engine.run_inference(input_tensor)
    
    print(f"  Raw hand_scores: {hand_scores}")
    print(f"  Raw handedness: {handedness}")
    print(f"  Landmarks shape: {landmarks.shape if landmarks is not None else None}")
    
    # Manual confidence analysis with detailed breakdown
    if landmarks is not None and len(landmarks) > 0:
        is_valid_detection, overall_confidence, analysis_details = analyzer.analyze_detection(
            landmarks[0], handedness, hand_scores, test_frame.shape
        )
        
        print(f"\n[DETAILED ANALYSIS]")
        print(f"  Analysis details: {analysis_details}")
        print(f"  Is valid detection: {is_valid_detection}")
        print(f"  Overall confidence: {overall_confidence:.3f}")
        
        # Check each signal individually
        presence_score = float(hand_scores[0][0])
        handedness_score = float(handedness[0][0])
        
        presence_ok = presence_score >= analyzer.presence_threshold
        handedness_ok = handedness_score >= analyzer.handedness_threshold
        
        print(f"\n[SIGNAL BREAKDOWN]")
        print(f"  1. Presence score: {presence_score:.3f} >= {analyzer.presence_threshold} = {presence_ok}")
        print(f"  2. Handedness score: {handedness_score:.3f} >= {analyzer.handedness_threshold} = {handedness_ok}")
        print(f"  3. Landmark consistency: {analysis_details['landmark_consistency']:.3f} >= {analyzer.landmark_consistency_threshold} = {analysis_details['landmark_consistency'] >= analyzer.landmark_consistency_threshold}")
        print(f"  4. Position validity: {analysis_details['position_validity']}")
        
        signals_passed = sum([
            presence_ok, 
            handedness_ok, 
            analysis_details['landmark_consistency'] >= analyzer.landmark_consistency_threshold,
            analysis_details['position_validity']
        ])
        print(f"  Total signals passed: {signals_passed}/4")
        
        # Check decision logic
        condition1 = presence_ok and signals_passed >= 3
        condition2 = presence_ok and (analysis_details['landmark_consistency'] >= analyzer.landmark_consistency_threshold) and presence_score > 0.6
        
        print(f"\n[DECISION LOGIC]")
        print(f"  Condition 1 (presence_ok AND signals_passed >= 3): {condition1}")
        print(f"  Condition 2 (presence_ok AND landmark_ok AND presence > 0.6): {condition2}")
        print(f"  Final decision: {condition1 or condition2}")
        
        # Analyze the landmarks themselves
        print(f"\n[LANDMARK ANALYSIS]")
        hand_landmarks = landmarks[0]
        if hand_landmarks.ndim == 1 and len(hand_landmarks) == 63:
            hand_landmarks = hand_landmarks.reshape(21, 3)
            
        print(f"  Landmark shape: {hand_landmarks.shape}")
        print(f"  Landmark range X: {hand_landmarks[:, 0].min():.1f} to {hand_landmarks[:, 0].max():.1f}")
        print(f"  Landmark range Y: {hand_landmarks[:, 1].min():.1f} to {hand_landmarks[:, 1].max():.1f}")
        print(f"  Landmark range Z: {hand_landmarks[:, 2].min():.3f} to {hand_landmarks[:, 2].max():.3f}")
        
        # Show first few landmarks
        print(f"  First 5 landmarks:")
        for i in range(5):
            print(f"    {i}: ({hand_landmarks[i][0]:.1f}, {hand_landmarks[i][1]:.1f}, {hand_landmarks[i][2]:.3f})")


def test_threshold_adjustment():
    """Test what happens with different presence thresholds."""
    print(f"\n[THRESHOLD TEST] Testing different presence thresholds...")
    
    # Initialize detection components
    model_loader = ModelLoader(model_path="/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
        
    inference_engine = InferenceEngine(model_loader)
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    # Test with gray background
    width, height = 640, 480
    test_frame = np.full((height, width, 3), (128, 128, 128), dtype=np.uint8)
    
    # Process frame
    processed_frame = frame_processor.preprocess(test_frame)
    input_tensor = inference_engine.prepare_input(processed_frame)
    landmarks, handedness, hand_scores = inference_engine.run_inference(input_tensor)
    
    presence_score = float(hand_scores[0][0])
    print(f"  Presence score for gray background: {presence_score:.3f}")
    
    # Test different thresholds
    test_thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for threshold in test_thresholds:
        analyzer = ConfidenceAnalyzer(presence_threshold=threshold)
        is_valid, confidence, details = analyzer.analyze_detection(
            landmarks[0], handedness, hand_scores, test_frame.shape
        )
        print(f"  Threshold {threshold:.1f}: Valid={is_valid}, Confidence={confidence:.3f}")


if __name__ == "__main__":
    analyze_false_positive()
    test_threshold_adjustment()
