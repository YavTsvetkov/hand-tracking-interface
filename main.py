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
from detection import ModelLoader, InferenceEngine
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
        print(f"[INFO] Loading palm detection model")
        # Load palm detection model
        self.model_loader = ModelLoader()
        if not self.model_loader.load_model():
            print("[ERROR] Failed to load palm detection model")
            sys.exit(1)
        # Initialize inference engine
        self.inference_engine = InferenceEngine(self.model_loader)
        
    def init_tracking(self):
        """Initialize tracking components optimized for raw position accuracy WITH false positive rejection."""
        self.hand_tracker = HandTracker(
            history_size=5,                             # Standard history buffer for better filtering
            detection_loss_frames=2,                    # Still quick to lose tracking for responsiveness
            stable_threshold=3                          # Require 3 frames to confirm detection (stronger false positive rejection)
        )
        
        self.position_validator = PositionValidator(
            max_speed=self.args.max_jump * 1.5,         # Lowered from 2x to 1.5x to catch unrealistic jumps
            min_speed=0.0,                              # Still accept completely stationary positions
            width=self.width,
            height=self.height
        )
        
        self.coordinate_smoother = CoordinateSmoother(
            smoothing_factor=0.0,                       # No explicit smoothing factor
            history_size=2                              # Still minimal history (2 frames) for real-time responsiveness
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
                landmarks, _, hand_scores, _ = self.inference_engine.run_inference(input_tensor)
                # Debug: palm detection confidence and normalized output
                # Convert normalized landmark to pixel wrist position
                if landmarks is not None and hand_scores is not None and hand_scores.size > 0:
                    norm_x, norm_y, _ = landmarks[0, 0]
                    wrist_position = (int(norm_x * self.width), int(norm_y * self.height))
                    confidence = float(hand_scores[0, 0])
                else:
                    wrist_position = None
                    confidence = 0.0
                print(f"[DEBUG] Palm center pixel: {wrist_position}, confidence: {confidence}")
                # Prepare landmarks list for rendering
                pixel_landmarks = [wrist_position] if wrist_position is not None else None
                
                # Initialize tracking variables
                is_valid = False
                is_suspicious = False
                
                # Update tracking
                if wrist_position is not None:
                    # Validate position physically
                    is_valid = self.position_validator.is_valid(wrist_position)
                    
                    # Skip suspiciously still detection at the beginning to establish tracking
                    if self.frame_count < 30:
                        is_suspicious = False
                    else:
                        # Check for suspiciously still position (false positive detection)
                        is_suspicious = self.position_validator.is_suspiciously_still(
                            threshold=self.args.false_positive_threshold
                        )
                        
                    if is_valid and not is_suspicious:
                        # Apply ultra-minimal smoothing optimized for raw accuracy
                        # Uses 90% current position, 10% previous for minimal jitter prevention
                        smoothed_position = self.coordinate_smoother.exponential_smooth(wrist_position)
                        
                                # Enhanced confidence checks to reject false positives
                        # Apply a stricter confidence threshold than before
                        confidence_threshold = self.args.confidence * 1.1  # Add 10% margin to be even more cautious
                        
                        if confidence > confidence_threshold:
                            # Check for stable tracking establishment - require higher confidence at the beginning
                            if not self.hand_tracker.is_hand_present and confidence < confidence_threshold * 1.2:
                                print(f"[DEBUG] Initial detection requires higher confidence. Got: {confidence:.2f}")
                                self.hand_tracker.update(None)
                            else:
                                # Update hand tracker with raw-optimized settings but enhanced false positive rejection
                                self.hand_tracker.update(smoothed_position, confidence)
                                print(f"[DEBUG] Updated tracker with near-raw position: {smoothed_position}, confidence: {confidence}")
                                if smoothed_position and wrist_position:
                                    diff_x = abs(smoothed_position[0] - wrist_position[0])
                                    diff_y = abs(smoothed_position[1] - wrist_position[1])
                                    print(f"[DEBUG] Raw vs smoothed diff: dx={diff_x:.1f}, dy={diff_y:.1f} px")
                        else:
                            # Low confidence detection - likely a false positive
                            print(f"[DEBUG] Rejecting low confidence detection: {confidence:.2f} < {confidence_threshold:.2f}")
                            self.hand_tracker.update(None)
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
                print(f"[DEBUG] Hand present: {is_hand_present}, quality: {tracking_quality:.2f}")
                
                # Display frame if not in headless mode
                if not self.headless:
                    # Render visualization
                    output_frame = self.frame_renderer.render_frame(
                        frame=frame,
                        landmarks=pixel_landmarks,
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
