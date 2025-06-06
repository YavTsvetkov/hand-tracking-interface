#!/usr/bin/env python3
"""
Debug script to analyze how landmark scores help with false positive detection.
"""

import numpy as np
import cv2
from detection.model_loader import ModelLoader
from detection.inference_engine import InferenceEngine
from detection.improved_confidence_analyzer import ConfidenceAnalyzer

def create_test_image(color, size=(224, 224)):
    """Create a solid color test image."""
    return np.full((size[1], size[0], 3), color, dtype=np.uint8)

def analyze_landmark_scores():
    """Analyze landmark scores for different backgrounds."""
    print("[DEBUG] Analyzing landmark scores behavior...")
    
    # Load model
    model_path = "/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite"
    model_loader = ModelLoader(model_path)
    model_loader.load_model()
    
    inference_engine = InferenceEngine(model_loader)
    analyzer = ConfidenceAnalyzer()
    
    # Test different backgrounds
    backgrounds = [
        ("BLACK", [0, 0, 0]),
        ("GRAY", [128, 128, 128]),
        ("RED", [255, 0, 0])
    ]
    
    for name, color in backgrounds:
        print(f"\n[{name} BACKGROUND]")
        test_frame = create_test_image(color)
        
        # Run inference
        input_tensor = inference_engine.prepare_input(test_frame)
        landmarks, handedness, hand_scores, landmark_scores = inference_engine.run_inference(input_tensor)
        
        print(f"  Presence score: {float(hand_scores[0][0]):.3f}")
        print(f"  Handedness: {float(handedness[0][0]):.3f}")
        
        # Analyze landmark scores
        scores = landmark_scores[0].reshape(21, 3)
        
        positive_count = np.sum(scores > 0)
        total_count = scores.size
        positive_ratio = positive_count / total_count
        score_variance = np.var(scores)
        mean_abs_score = np.mean(np.abs(scores))
        
        # Key landmarks analysis
        key_landmarks = [0, 4, 8, 12, 16, 20]  # wrist + fingertips
        key_scores = scores[key_landmarks].flatten()
        key_positive_ratio = np.sum(key_scores > 0) / len(key_scores)
        
        print(f"  Landmark scores analysis:")
        print(f"    Positive ratio: {positive_ratio:.3f} ({positive_count}/{total_count})")
        print(f"    Score variance: {score_variance:.6f}")
        print(f"    Mean abs score: {mean_abs_score:.6f}")
        print(f"    Key landmarks positive: {key_positive_ratio:.3f}")
        
        # Test the new analysis function
        is_valid_scores = analyzer._analyze_landmark_scores(landmark_scores[0])
        print(f"    Landmark scores valid: {is_valid_scores}")
        
        # Show some sample scores
        print(f"    Sample scores (first 10): {scores.flatten()[:10]}")

if __name__ == "__main__":
    analyze_landmark_scores()
