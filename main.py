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
from ros_cmd_vel import CoordinateParser


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
        self.init_coordinate_parser()
        
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
        
    def init_coordinate_parser(self):
        """Initialize coordinate parser for cmd_vel control."""
        print(f"[INFO] Initializing coordinate parser")
        self.coordinate_parser = CoordinateParser(
            frame_width=self.width,
            frame_height=self.height,
            max_linear_speed=0.5,
            max_angular_speed=1.0
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
                processed_frame, crop_offset_x, crop_offset_y = self.frame_processor.preprocess(frame)
                processed_frame_height, processed_frame_width = processed_frame.shape[:2]
                
                # Set frame dimensions and crop offsets in InferenceEngine
                self.inference_engine.set_frame_dimensions(
                    frame_shape=(self.height, self.width),
                    processed_frame_shape=(processed_frame_height, processed_frame_width),
                    crop_offset_x=crop_offset_x,
                    crop_offset_y=crop_offset_y
                )

                # Prepare input for model
                input_tensor = self.inference_engine.prepare_input(processed_frame)
                
                # Run inference
                landmarks, _, hand_scores, _ = self.inference_engine.run_inference(input_tensor)
                wrist_position = None
                norm_wrist_pos_on_processed_frame = None # For debug display

                if landmarks is not None and hand_scores is not None and hand_scores[0][0] > self.args.confidence:
                    # Landmarks are normalized relative to the processed_frame (after crop, before letterbox)
                    norm_x = landmarks[0][0][0] 
                    norm_y = landmarks[0][0][1]
                    norm_wrist_pos_on_processed_frame = (norm_x, norm_y) # Store for debug

                    # Convert normalized coordinates (relative to processed_frame) to pixel coordinates 
                    # on the *original* camera frame.
                    # Step 1: Scale to processed_frame dimensions
                    pixel_x_on_processed = norm_x * processed_frame_width
                    pixel_y_on_processed = norm_y * processed_frame_height
                    
                    # Step 2: Add crop offsets to map to original frame
                    pixel_x_on_original = pixel_x_on_processed + crop_offset_x
                    pixel_y_on_original = pixel_y_on_processed + crop_offset_y
                    
                    # Ensure coordinates are within original frame bounds
                    wrist_position = (
                        int(np.clip(pixel_x_on_original, 0, self.width - 1)),
                        int(np.clip(pixel_y_on_original, 0, self.height - 1))
                    )
                    
                    # Debug print for coordinate transformation
                    if self.args.debug:
                        print(f"[MAIN_DEBUG] Norm coords (on proc_frame): ({norm_x:.3f}, {norm_y:.3f})")
                        print(f"[MAIN_DEBUG] Processed frame shape: ({processed_frame_width}, {processed_frame_height})")
                        print(f"[MAIN_DEBUG] Crop offsets (x,y): ({crop_offset_x}, {crop_offset_y})")
                        print(f"[MAIN_DEBUG] Pixel (on proc_frame): ({pixel_x_on_processed:.1f}, {pixel_y_on_processed:.1f})")
                        print(f"[MAIN_DEBUG] Final Wrist Pos (on orig_frame): {wrist_position}")

                    # Validate and smooth coordinates
                    if self.position_validator.is_valid(wrist_position):
                        wrist_position = self.coordinate_smoother.exponential_smooth(wrist_position) # Changed from smooth
                    else:
                        wrist_position = None # Invalidate if jump is too large
                else:
                    # No valid detection or low confidence
                    self.coordinate_smoother.reset() # Reset smoother if hand is lost

                # Update status display
                if self.args.debug: # Changed from self.args.display_debug_text
                    self.status_display.update_value("FPS", f"{self.fps_counter.get_fps():.2f}")
                    if wrist_position:
                        self.status_display.update_value("Wrist Pos", f"({wrist_position[0]},{wrist_position[1]})")
                    else:
                        self.status_display.update_value("Wrist Pos", "N/A")
                    self.status_display.draw(frame)
                
                # Calculate cmd_vel values for display
                cmd_vel_values = None
                if wrist_position is not None:
                    # The get_cmd_vel_display_values method in CoordinateParser
                    # uses the frame dimensions provided during its initialization.
                    # It does not need them passed again here.
                    cmd_vel_values = self.coordinate_parser.get_cmd_vel_display_values(wrist_position)
                else:
                    cmd_vel_values = self.coordinate_parser.get_cmd_vel_display_values(None)
                
                print(f"[DEBUG] Hand present: {wrist_position is not None}")
                if cmd_vel_values:
                    print(f"[DEBUG] Cmd_vel: linear={cmd_vel_values['linear_raw']:.2f}, angular={cmd_vel_values['angular_raw']:.2f}")
                    print(f"[DEBUG] Display values: linear={cmd_vel_values['linear_display']:.1f}, angular={cmd_vel_values['angular_display']:.1f}")
                
                # Display frame if not in headless mode
                if not self.headless:
                    # Render visualization
                    output_frame = self.frame_renderer.render_frame(
                        frame=frame,
                        landmarks=[wrist_position] if wrist_position is not None else None,
                        wrist_position=wrist_position,
                        tracking_quality=1.0,  # Tracking quality not available in this context
                        is_hand_present=wrist_position is not None,
                        is_valid=True,  # Always true here, validation is done above
                        fps=self.fps_counter.get_fps(),
                        cmd_vel_values=cmd_vel_values,
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
                        status = "DETECTED" if wrist_position is not None else "NOT DETECTED"
                        print(f"[STATUS] Hand: {status}, FPS: {self.fps_counter.get_fps():.1f}")
                        
                        if wrist_position is not None:
                            print(f"[POSITION] Wrist at: {wrist_position}")
                
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
