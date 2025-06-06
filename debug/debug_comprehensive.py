#!/usr/bin/env python3
"""
Comprehensive debug script to investigate hand detection logic.
Tests with real camera feed, solid backgrounds, and detailed step-by-step analysis.
"""

import cv2
import time
import sys
import numpy as np

# Import modules
from config.settings import Settings
from camera import LibcameraCapture, FrameProcessor
from detection import ModelLoader, InferenceEngine, LandmarkExtractor
from tracking import HandTracker


def test_solid_background(model_loader, inference_engine, landmark_extractor, color=(0, 0, 0), color_name="BLACK"):
    """Test detection with solid color background."""
    print(f"\n[TEST] Testing with {color_name} background...")
    
    # Create solid color image
    width, height = 640, 480
    test_frame = np.full((height, width, 3), color, dtype=np.uint8)
    
    # Process frame
    frame_processor = FrameProcessor(crop_factor=0.8)
    processed_frame = frame_processor.preprocess(test_frame)
    
    print(f"[DEBUG] Test frame shape: {test_frame.shape}")
    print(f"[DEBUG] Processed frame shape: {processed_frame.shape}")
    print(f"[DEBUG] Test frame pixel values (sample): {test_frame[100:105, 100:105, :]}")
    
    # Prepare input for model
    input_tensor = inference_engine.prepare_input(processed_frame)
    print(f"[DEBUG] Input tensor shape: {input_tensor.shape}")
    print(f"[DEBUG] Input tensor range: {input_tensor.min():.3f} to {input_tensor.max():.3f}")
    print(f"[DEBUG] Input tensor sample values: {input_tensor[0, 100:105, 100:105, 0]}")
    
    # Run inference
    landmarks, handedness, hand_scores = inference_engine.run_inference(input_tensor)
    
    print(f"[DEBUG] Raw model outputs:")
    print(f"  landmarks shape: {landmarks.shape if landmarks is not None else None}")
    print(f"  handedness shape: {handedness.shape if handedness is not None else None}")
    print(f"  hand_scores shape: {hand_scores.shape if hand_scores is not None else None}")
    print(f"  hand_scores value: {hand_scores}")
    print(f"  handedness value: {handedness}")
    
    # Extract landmarks with detailed debugging
    pixels, wrist_position, confidence = landmark_extractor.extract_landmarks(
        landmarks, handedness, hand_scores, test_frame.shape
    )
    
    print(f"[DEBUG] Landmark extraction results:")
    print(f"  pixels: {pixels is not None}")
    print(f"  wrist_position: {wrist_position}")
    print(f"  confidence: {confidence}")
    
    # Save test image for visual verification
    cv2.imwrite(f"/home/rtsvetkov/hand_tracking/test_{color_name.lower()}_bg.jpg", test_frame)
    print(f"[DEBUG] Saved test image: test_{color_name.lower()}_bg.jpg")
    
    return wrist_position is not None, confidence


def debug_confidence_analyzer():
    """Test the confidence analyzer directly."""
    print("\n[TEST] Testing confidence analyzer logic...")
    
    # Import confidence analyzer
    from detection.improved_confidence_analyzer import ConfidenceAnalyzer
    
    analyzer = ConfidenceAnalyzer()
    
    # Test with different presence scores
    test_cases = [
        (0.1, "Very low presence"),
        (0.25, "Low presence (near threshold)"),
        (0.3, "At threshold"),
        (0.5, "Medium presence"),
        (0.8, "High presence"),
    ]
    
    # Mock data
    fake_landmarks = np.random.random((21, 3)) * 100  # Random landmarks
    fake_handedness = np.array([[0.5]])  # Neutral handedness
    frame_shape = (480, 640)
    
    for presence_score, description in test_cases:
        fake_hand_scores = np.array([[presence_score]])
        
        is_valid, overall_conf, details = analyzer.analyze_detection(
            fake_landmarks, fake_handedness, fake_hand_scores, frame_shape
        )
        
        print(f"[TEST] {description}: presence={presence_score:.3f} -> valid={is_valid}, conf={overall_conf:.3f}")
        print(f"       Details: {details}")


def debug_detection_comprehensive():
    """Comprehensive debugging of the detection pipeline."""
    print("[DEBUG] === COMPREHENSIVE HAND DETECTION DEBUG ===")
    
    # Initialize components
    print("[DEBUG] Initializing components...")
    
    # Initialize detection components
    model_loader = ModelLoader(model_path="/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
        
    inference_engine = InferenceEngine(model_loader)
    landmark_extractor = LandmarkExtractor(confidence_threshold=0.2)
    
    # Test confidence analyzer logic
    debug_confidence_analyzer()
    
    # Test with solid backgrounds
    print("\n[DEBUG] === TESTING WITH SOLID BACKGROUNDS ===")
    
    # Test black background
    black_detected, black_conf = test_solid_background(
        model_loader, inference_engine, landmark_extractor, 
        color=(0, 0, 0), color_name="BLACK"
    )
    
    # Test white background  
    white_detected, white_conf = test_solid_background(
        model_loader, inference_engine, landmark_extractor,
        color=(255, 255, 255), color_name="WHITE"
    )
    
    # Test gray background
    gray_detected, gray_conf = test_solid_background(
        model_loader, inference_engine, landmark_extractor,
        color=(128, 128, 128), color_name="GRAY"
    )
    
    print(f"\n[RESULTS] Solid background test results:")
    print(f"  BLACK background: detected={black_detected}, confidence={black_conf:.3f}")
    print(f"  WHITE background: detected={white_detected}, confidence={white_conf:.3f}")
    print(f"  GRAY background: detected={gray_detected}, confidence={gray_conf:.3f}")
    
    if black_detected or white_detected or gray_detected:
        print("[WARNING] ⚠️  FALSE POSITIVE DETECTION on solid background!")
        print("[WARNING] This suggests the model or confidence logic has issues")
    else:
        print("[INFO] ✅ No false positives on solid backgrounds - good!")
    
    # Test with real camera
    print("\n[DEBUG] === TESTING WITH REAL CAMERA ===")
    debug_real_camera(model_loader, inference_engine, landmark_extractor)


def debug_real_camera(model_loader, inference_engine, landmark_extractor):
    """Debug with real camera feed."""
    
    # Initialize camera
    width, height = 640, 480
    camera = LibcameraCapture(width, height, fps=10)
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    # Initialize tracker
    hand_tracker = HandTracker(detection_loss_frames=5, stable_threshold=3)
    
    print("[DEBUG] Starting real camera test...")
    print("[DEBUG] Instructions:")
    print("  1. First, cover the camera completely (should show NO HAND)")
    print("  2. Then point camera at empty wall/background (should show NO HAND)")
    print("  3. Finally, show your hand (should show HAND DETECTED)")
    print("  Press 'q' to quit")
    
    frame_count = 0
    
    try:
        while True:
            frame_count += 1
            
            # Capture frame
            ret, frame = camera.read()
            if not ret or frame is None:
                print("[WARNING] Failed to capture frame")
                time.sleep(0.01)
                continue
            
            print(f"\n[FRAME {frame_count:03d}] === STEP-BY-STEP DEBUG ===")
            
            # Step 1: Frame analysis
            print(f"[STEP 1] Frame analysis:")
            print(f"  Frame shape: {frame.shape}")
            print(f"  Frame dtype: {frame.dtype}")
            print(f"  Frame range: {frame.min()} to {frame.max()}")
            print(f"  Frame mean: {frame.mean():.1f}")
            print(f"  Frame std: {frame.std():.1f}")
            
            # Step 2: Preprocessing
            print(f"[STEP 2] Preprocessing:")
            processed_frame = frame_processor.preprocess(frame)
            print(f"  Processed shape: {processed_frame.shape}")
            print(f"  Processed range: {processed_frame.min()} to {processed_frame.max()}")
            print(f"  Processed mean: {processed_frame.mean():.1f}")
            
            # Step 3: Model input preparation
            print(f"[STEP 3] Model input preparation:")
            input_tensor = inference_engine.prepare_input(processed_frame)
            print(f"  Input tensor shape: {input_tensor.shape}")
            print(f"  Input tensor range: {input_tensor.min():.3f} to {input_tensor.max():.3f}")
            print(f"  Input tensor mean: {input_tensor.mean():.3f}")
            
            # Step 4: Model inference
            print(f"[STEP 4] Model inference:")
            landmarks, handedness, hand_scores = inference_engine.run_inference(input_tensor)
            print(f"  Raw landmarks shape: {landmarks.shape if landmarks is not None else None}")
            print(f"  Raw handedness: {handedness}")
            print(f"  Raw hand_scores: {hand_scores}")
            print(f"  Raw presence score: {float(hand_scores[0][0]) if hand_scores is not None else 'None'}")
            
            # Step 5: Confidence analysis
            print(f"[STEP 5] Confidence analysis:")
            pixels, wrist_position, confidence = landmark_extractor.extract_landmarks(
                landmarks, handedness, hand_scores, frame.shape
            )
            print(f"  Landmarks extracted: {pixels is not None}")
            print(f"  Wrist position: {wrist_position}")
            print(f"  Final confidence: {confidence:.3f}")
            
            # Step 6: Tracking update
            print(f"[STEP 6] Tracking update:")
            if wrist_position is not None:
                result = hand_tracker.update(wrist_position, confidence)
                print(f"  Updated tracker with position: {result}")
            else:
                result = hand_tracker.update(None)
                print(f"  Updated tracker with no detection: {result}")
            
            # Step 7: Final state
            print(f"[STEP 7] Final tracking state:")
            is_hand_present = hand_tracker.is_hand_present
            print(f"  Hand present: {is_hand_present}")
            print(f"  Consecutive detections: {hand_tracker.consecutive_detections}")
            print(f"  Consecutive non-detections: {hand_tracker.consecutive_non_detections}")
            print(f"  Tracking quality: {hand_tracker.get_tracking_quality():.3f}")
            
            # Visual display
            display_frame = frame.copy()
            
            # Draw wrist if detected
            if wrist_position is not None:
                cv2.circle(display_frame, wrist_position, 10, (0, 255, 0), -1)
                cv2.putText(display_frame, f"Conf: {confidence:.2f}", 
                           (wrist_position[0] + 15, wrist_position[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Status text
            status_text = "HAND DETECTED" if is_hand_present else "NO HAND"
            status_color = (0, 255, 0) if is_hand_present else (0, 0, 255)
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            # Additional debug info
            cv2.putText(display_frame, f"Raw: {float(hand_scores[0][0]):.3f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Frame: {frame_count}", (10, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Comprehensive Debug", display_frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
            # Pause between frames for readability
            time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_detection_comprehensive()
