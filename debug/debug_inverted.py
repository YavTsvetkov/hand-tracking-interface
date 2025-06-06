#!/usr/bin/env python3
"""
Debug script to investigate the inverted hand detection issue.
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


def debug_detection():
    """Debug the detection pipeline to find inverted logic."""
    print("[DEBUG] Initializing components...")
    
    # Initialize camera
    width, height = 640, 480
    camera = LibcameraCapture(width, height, fps=10)
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    # Initialize detection
    model_loader = ModelLoader(model_path="/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
        
    inference_engine = InferenceEngine(model_loader)
    landmark_extractor = LandmarkExtractor(confidence_threshold=0.2)
    
    # Initialize tracker
    hand_tracker = HandTracker(detection_loss_frames=5, stable_threshold=3)
    
    print("[DEBUG] Starting capture loop...")
    print("[DEBUG] Instructions: Hold your hand in view, then remove it")
    print("[DEBUG] Press 'q' to quit")
    
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
            
            # Process frame
            processed_frame = frame_processor.preprocess(frame)
            
            # Prepare input for model
            input_tensor = inference_engine.prepare_input(processed_frame)
            
            # Run inference
            landmarks, handedness, hand_scores = inference_engine.run_inference(input_tensor)
            
            # Extract landmarks
            pixels, wrist_position, confidence = landmark_extractor.extract_landmarks(
                landmarks, handedness, hand_scores, frame.shape
            )
            
            # Update tracking
            if wrist_position is not None:
                hand_tracker.update(wrist_position, confidence)
            else:
                hand_tracker.update(None)
            
            # Get tracker state
            is_hand_present = hand_tracker.is_hand_present
            
            # Print detailed debug info every 10 frames
            if frame_count % 10 == 0:
                print(f"\n[FRAME {frame_count:03d}] DETAILED DEBUG:")
                print(f"  Raw hand_scores: {hand_scores}")
                print(f"  Raw handedness: {handedness}")
                print(f"  Wrist position: {wrist_position}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Hand tracker state: {is_hand_present}")
                print(f"  Consecutive detections: {hand_tracker.consecutive_detections}")
                print(f"  Consecutive non-detections: {hand_tracker.consecutive_non_detections}")
                
                # Manual visual assessment
                if wrist_position is not None:
                    print(f"  VISUAL CHECK: Can you see a hand in the image? (Detection found)")
                else:
                    print(f"  VISUAL CHECK: Can you see a hand in the image? (No detection)")
                    
                if is_hand_present:
                    print(f"  TRACKER SAYS: HAND DETECTED")
                else:
                    print(f"  TRACKER SAYS: NO HAND")
                    
                print("-" * 50)
            
            # Simple visual display
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
            
            # Show frame
            cv2.imshow("Debug Detection", display_frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_detection()
