#!/usr/bin/env python3
"""
Main entry point for the hand tracking application.
This module coordinates all components of the hand tracking system.
"""

import cv2
import time
import sys
import signal
import numpy as np

# Import modules
from config.settings import Settings
from camera import LibcameraCapture, FrameProcessor
from detection import ModelLoader, InferenceEngine, LandmarkExtractor
from tracking import HandTracker, PositionValidator, CoordinateSmoother
from visualization import Renderer, SimpleStatusDisplay, FrameRenderer
from utils import parse_args, FPSCounter


class HandTrackingApp:
    """Main hand tracking application."""
    
    def __init__(self):
        """Initialize the hand tracking application."""
        # Parse command line arguments
        self.args = parse_args()
        
        # Initialize components
        self.init_camera()
        self.init_detection()
        self.init_tracking()
        self.init_visualization()
        
        # Performance tracking
        self.fps_counter = FPSCounter(window_size=30)
        self.frame_count = 0
        self.frame_skip = self.args.frame_skip
        
        # Register signal handlers for clean shutdown
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        
        # Running state
        self.running = True
        
    def init_camera(self):
        """Initialize camera components."""
        # Parse resolution
        width, height = Settings.get_resolution_as_tuple(self.args.res)
        self.width = width
        self.height = height
        
        # Initialize camera
        print(f"[INFO] Initializing camera ({width}x{height} @ {self.args.fps} FPS)")
        self.camera = LibcameraCapture(width, height, fps=self.args.fps)
        self.frame_processor = FrameProcessor(crop_factor=self.args.crop_factor)
        
    def init_detection(self):
        """Initialize detection components."""
        print(f"[INFO] Loading model: models/hand_landmark_lite.tflite")
        self.model_loader = ModelLoader()  # Fixed model path
        if not self.model_loader.load_model():
            print("[ERROR] Failed to load model")
            sys.exit(1)
            
        self.inference_engine = InferenceEngine(self.model_loader)
        self.landmark_extractor = LandmarkExtractor(confidence_threshold=self.args.confidence)
        
    def init_tracking(self):
        """Initialize tracking components."""
        self.hand_tracker = HandTracker(
            detection_loss_frames=self.args.detection_loss_frames,
            stable_threshold=1  # Reduced frames needed to confirm detection for immediate preview
        )
        
        self.position_validator = PositionValidator(
            max_speed=self.args.max_jump,
            min_speed=0.5,
            width=self.width,
            height=self.height
        )
        
        self.coordinate_smoother = CoordinateSmoother(
            smoothing_factor=self.args.smoothing,
            history_size=5
        )
        
    def init_visualization(self):
        """Initialize visualization components."""
        self.headless = self.args.headless
        
        if not self.headless:
            self.renderer = Renderer()
            self.status_display = SimpleStatusDisplay(self.width, self.height)
            self.frame_renderer = FrameRenderer(
                width=self.width, 
                height=self.height,
                debug_mode=self.args.debug
            )
        
    def run_detection_loop(self):
        """Run the main detection loop."""
        print("[INFO] Starting detection loop")
        
        try:
            while self.running:
                # Update FPS counter
                self.fps_counter.update()
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    print("[WARNING] Failed to capture frame")
                    time.sleep(0.01)
                    continue
                
                # Skip frames for performance if needed
                self.frame_count += 1
                if self.frame_skip > 1 and self.frame_count % self.frame_skip != 0:
                    # Just show the frame without processing
                    if not self.headless:
                        cv2.imshow("Hand Tracking", frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue
                
                # Process frame
                processed_frame = self.frame_processor.preprocess(frame)
                
                # Prepare input for model
                input_tensor = self.inference_engine.prepare_input(processed_frame)
                
                # Run inference
                landmarks, handedness, hand_scores, landmark_scores = self.inference_engine.run_inference(input_tensor)
                print(f"[DEBUG] Inference output - hand_scores: {hand_scores}, landmark_scores: {landmark_scores}")
                
                # Extract landmarks
                pixels, wrist_position, confidence = self.landmark_extractor.extract_landmarks(
                    landmarks, handedness, hand_scores, landmark_scores, frame.shape
                )
                print(f"[DEBUG] Landmark extraction - wrist_position: {wrist_position}, confidence: {confidence}")
                
                # Initialize tracking variables
                is_valid = False
                is_suspicious = False
                
                # Update tracking
                if wrist_position is not None:
                    # Validate position physically
                    is_valid = self.position_validator.is_valid(wrist_position)
                    print(f"[DEBUG] PositionValidator.is_valid -> {is_valid} for wrist_position {wrist_position}")
                    
                    # Skip suspiciously still detection at the beginning to establish tracking
                    if self.frame_count < 30:
                        is_suspicious = False
                    else:
                        # Check for suspiciously still position (false positive detection)
                        is_suspicious = self.position_validator.is_suspiciously_still(
                            threshold=self.args.false_positive_threshold
                        )
                        print(f"[DEBUG] PositionValidator.is_suspiciously_still -> {is_suspicious}")
                    
                    if is_valid and not is_suspicious:
                        # Apply smoothing
                        smoothed_position = self.coordinate_smoother.exponential_smooth(wrist_position)
                        
                        # Update hand tracker
                        self.hand_tracker.update(smoothed_position, confidence)
                        print(f"[DEBUG] Updated tracker with smoothed_position: {smoothed_position}, confidence: {confidence}")
                    else:
                        # Position is invalid or suspicious, don't update tracking
                        wrist_position = None
                        self.hand_tracker.update(None)
                        print(f"[DEBUG] Updated tracker with None - is_valid: {is_valid}, is_suspicious: {is_suspicious}")
                else:
                    # No detection
                    self.hand_tracker.update(None)
                    
                # Get tracking state
                is_hand_present = self.hand_tracker.is_hand_present
                tracking_quality = self.hand_tracker.get_tracking_quality()
                print(f"[DEBUG] HandTracker state - is_hand_present: {is_hand_present}, tracking_quality: {tracking_quality:.2f}")
                
                # Display frame if not in headless mode
                if not self.headless:
                    # Render visualization
                    output_frame = self.frame_renderer.render_frame(
                        frame=frame,
                        landmarks=pixels,
                        wrist_position=wrist_position,
                        tracking_quality=tracking_quality,
                        is_hand_present=is_hand_present,
                        is_valid=not (is_suspicious),
                        fps=self.fps_counter.get_fps(),
                        renderer=self.renderer,
                        status_display=self.status_display
                    )
                    
                    # Display frame
                    cv2.imshow("Hand Tracking", output_frame)
                    
                    # Check for key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('r'):
                        # Reset tracking
                        self.hand_tracker.reset()
                        self.position_validator.reset()
                        self.coordinate_smoother.reset()
                else:
                    # Headless mode - just print status occasionally
                    if self.frame_count % 30 == 0:
                        status = "DETECTED" if is_hand_present else "NOT DETECTED"
                        print(f"[STATUS] Hand: {status}, FPS: {self.fps_counter.get_fps():.1f}")
                        
                        if wrist_position is not None:
                            print(f"[POSITION] Wrist at: {wrist_position}, Quality: {tracking_quality:.2f}")
                
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user")
        finally:
            self.cleanup()
            
    def cleanup(self, signum=None, frame=None):
        """Clean up resources."""
        if not self.running:
            return
            
        self.running = False
        print("[INFO] Cleaning up resources...")
        
        # Release camera
        if hasattr(self, 'camera'):
            self.camera.release()
            
        # Close all OpenCV windows
        cv2.destroyAllWindows()
            

if __name__ == "__main__":
    app = HandTrackingApp()
    app.run_detection_loop()
