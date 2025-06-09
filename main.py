#!/usr/bin/env python3
"""
Improved main entry point for hand tracking with simplified coordinate extraction.
Removes complex transformations while maintaining accuracy.
"""

import cv2
import time
import sys
import signal
import numpy as np

# Import existing modules where needed
from config.settings import Settings
from config.ros_config import RosConfig
from camera import LibcameraCapture
from utils import parse_args, FPSCounter
from ros_cmd_vel import CoordinateParser
from config.ros_config import RosConfig

# Try to import ROS 2 components (optional)
try:
    from ros_cmd_vel.ros_publisher import RosPublisher
    import rclpy
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False
    print("[WARNING] ROS 2 not available. Running in simulation mode.")

# Simple MediaPipe-based detector
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not available. Using existing palm detection.")


class HandTracker:
    """Hand tracking using MediaPipe or existing palm detection."""
    
    def __init__(self, use_mediapipe=True, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.use_mediapipe = use_mediapipe and MEDIAPIPE_AVAILABLE
        
        if self.use_mediapipe:
            self._init_mediapipe()
        else:
            self._init_palm_detection()
        
        # Simple smoothing
        self.prev_center = None
        self.smoothing_factor = RosConfig.HAND_SMOOTHING_FACTOR
        
    def _init_mediapipe(self):
        """Initialize MediaPipe hands detection."""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def _init_palm_detection(self):
        """Initialize existing palm detection system."""
        from detection import ModelLoader, InferenceEngine
        from camera import FrameProcessor
        
        print("[INFO] Using existing palm detection model")
        self.model_loader = ModelLoader()
        if not self.model_loader.load_model():
            print("[ERROR] Failed to load palm detection model")
            sys.exit(1)
        
        self.inference_engine = InferenceEngine(self.model_loader)
        self.frame_processor = FrameProcessor(crop_factor=0.8)  # Simple crop
        
    def detect_hand_center(self, frame):
        """Detect hand center using the simpler approach."""
        if self.use_mediapipe:
            return self._detect_with_mediapipe(frame)
        else:
            return self._detect_with_palm_model(frame)
    
    def _detect_with_mediapipe(self, frame):
        """Detect hand center using MediaPipe."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate center from all landmarks
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            
            # Convert normalized coordinates to pixels
            norm_center_x = np.mean(x_coords)
            norm_center_y = np.mean(y_coords)
            center_x = int(norm_center_x * self.camera_width)
            center_y = int(norm_center_y * self.camera_height)
            
            return (center_x, center_y)
        
        return None
    
    def _detect_with_palm_model(self, frame):
        """Detect hand center using existing palm detection model (simplified)."""
        # Simple preprocessing - just crop and resize
        processed_frame, crop_offset_x, crop_offset_y = self.frame_processor.preprocess(frame)
        processed_height, processed_width = processed_frame.shape[:2]
        
        # Set frame dimensions
        self.inference_engine.set_frame_dimensions(
            frame_shape=(self.camera_height, self.camera_width),
            processed_frame_shape=(processed_height, processed_width),
            crop_offset_x=crop_offset_x,
            crop_offset_y=crop_offset_y
        )
        
        # Prepare input and run inference
        input_tensor = self.inference_engine.prepare_input(processed_frame)
        landmarks, _, hand_scores, _ = self.inference_engine.run_inference(input_tensor)
        
        if landmarks is not None and hand_scores is not None and hand_scores[0][0] > 0.5:
            # Get the first landmark (palm center) - simplified extraction
            norm_x = landmarks[0][0][0]  # Normalized X
            norm_y = landmarks[0][0][1]  # Normalized Y
            
            # Convert to pixel coordinates with simple scaling
            pixel_x = int(norm_x * processed_width) + crop_offset_x
            pixel_y = int(norm_y * processed_height) + crop_offset_y
            
            # Clamp to frame bounds
            pixel_x = np.clip(pixel_x, 0, self.camera_width - 1)
            pixel_y = np.clip(pixel_y, 0, self.camera_height - 1)
            
            return (pixel_x, pixel_y)
        
        return None
    
    def process_frame(self, frame):
        """Process frame and return smoothed hand center."""
        # Flip frame horizontally to remove mirror effect
        frame = cv2.flip(frame, 1)
        
        hand_center = self.detect_hand_center(frame)
        
        if hand_center is not None:
            # Convert hand coordinates back to unflipped coordinate system
            # Since we flipped the frame, we need to flip the x-coordinate back
            unflipped_x = self.camera_width - 1 - hand_center[0]
            hand_center = (unflipped_x, hand_center[1])
            
            # Simple smoothing
            if self.prev_center is not None:
                smoothed_x = int(self.prev_center[0] + self.smoothing_factor * (hand_center[0] - self.prev_center[0]))
                smoothed_y = int(self.prev_center[1] + self.smoothing_factor * (hand_center[1] - self.prev_center[1]))
                hand_center = (smoothed_x, smoothed_y)
            
            self.prev_center = hand_center
        else:
            # Reset smoothing when hand is lost
            self.prev_center = None
            
        return hand_center


class CartesianController:
    """4-quadrant Cartesian controller for cmd_vel with ROS integration."""
    
    def __init__(self, frame_width=640, frame_height=480, dead_zone=50):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        self.dead_zone = dead_zone
        
        # Initialize ROS coordinate parser with proper configuration
        ros_config = RosConfig.get_parser_config(frame_width, frame_height)
        self.ros_parser = CoordinateParser(
            frame_width=ros_config['frame_width'],
            frame_height=ros_config['frame_height'],
            max_linear_speed=ros_config['max_linear_speed'],
            max_angular_speed=ros_config['max_angular_speed'],
            dead_zone_radius=ros_config['dead_zone_radius'],
            control_zone_radius=ros_config['control_zone_radius']
        )
        
        # Initialize ROS publisher if available
        self.ros_publisher = None
        self.ros_available = ROS_AVAILABLE  # Store global variable as instance variable
        if self.ros_available:
            try:
                rclpy.init()
                ros_config = RosConfig.get_ros_config()
                self.ros_publisher = RosPublisher(
                    node_name=ros_config['node_name'],
                    topic_name=ros_config['topic_name'],
                    publish_rate=ros_config['publish_rate']
                )
            except Exception as e:
                self.ros_available = False
        
        # Maximum speeds (for simple display)
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        
    def get_cmd_vel(self, hand_position):
        """Convert hand position to cmd_vel commands with ROS integration."""
        if hand_position is None:
            ros_cmd = {'linear_x': 0.0, 'angular_z': 0.0}
        else:
            # Use ROS coordinate parser for proper conversion
            ros_cmd = self.ros_parser.parse_palm_coordinates(hand_position)
        
        # Send to ROS 2 if available
        if self.ros_publisher is not None:
            try:
                self.ros_publisher.update_command(
                    linear_x=ros_cmd['linear_x'],
                    angular_z=ros_cmd['angular_z']
                )
            except Exception as e:
                pass  # Silently handle ROS errors in production
        
        # Calculate additional display info for compatibility
        if hand_position is None:
            return {
                'linear_x': 0.0, 
                'angular_z': 0.0, 
                'zone': 'none',
                'linear_display': 0.0,
                'angular_display': 0.0,
                'linear_raw': 0.0,
                'angular_raw': 0.0
            }
        
        x, y = hand_position
        
        # Calculate offset from center for zone detection
        dx = x - self.center_x
        dy = self.center_y - y  # Flip Y axis for proper Cartesian coordinates (positive Y = up)
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Determine zone
        zone = self._get_zone(dx, dy) if distance >= self.dead_zone else 'center'
        
        return {
            'linear_x': ros_cmd['linear_x'],
            'angular_z': ros_cmd['angular_z'],
            'zone': zone,
            'intensity': distance / (min(self.frame_width, self.frame_height) / 2),
            'linear_display': ros_cmd['linear_x'] * 10,  # For display
            'angular_display': ros_cmd['angular_z'] * 10,  # For display  
            'linear_raw': ros_cmd['linear_x'],
            'angular_raw': ros_cmd['angular_z']
        }
    
    def _get_zone(self, dx, dy):
        """Determine which quadrant/zone the hand is in using proper Cartesian coordinates."""
        # dx: negative = left, positive = right
        # dy: negative = backward/down, positive = forward/up
        
        if abs(dx) > abs(dy):
            # Horizontal movement dominates
            return 'right' if dx > 0 else 'left'
        else:
            # Vertical movement dominates
            return 'forward' if dy > 0 else 'backward'


class HandTrackingApp:
    """Hand tracking application with Cartesian coordinate system."""
    
    def __init__(self):
        """Initialize the hand tracking application."""
        # Parse command line arguments
        self.args = parse_args()
        
        # Initialize camera
        self.init_camera()
        
        # Initialize simplified tracker
        self.hand_tracker = HandTracker(
            use_mediapipe=MEDIAPIPE_AVAILABLE,
            camera_width=self.width,
            camera_height=self.height
        )
        
        # Initialize simple controller
        self.controller = CartesianController(
            frame_width=self.width,
            frame_height=self.height,
            dead_zone=50
        )
        
        # Performance tracking
        self.fps_counter = FPSCounter(window_size=30)
        self.frame_count = 0
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)
        self.running = True
        
    def init_camera(self):
        """Initialize camera components."""
        width, height = Settings.get_resolution_as_tuple(self.args.res)
        self.width = width
        self.height = height
        
        self.camera = LibcameraCapture(width, height, fps=self.args.fps)
        
    def run_detection_loop(self):
        """Run the simplified detection loop."""
        print("[INFO] Starting hand tracking system")
        print(f"[INFO] Detection method: {'MediaPipe' if MEDIAPIPE_AVAILABLE else 'Palm Detection Model'}")
        print(f"[INFO] ROS integration: {'Enabled' if self.controller.ros_available else 'Disabled'}")
        
        try:
            while self.running:
                self.fps_counter.update()
                
                # Capture frame
                ret, frame = self.camera.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Process frame
                hand_center = self.hand_tracker.process_frame(frame)
                
                # Get cmd_vel values
                cmd_vel = self.controller.get_cmd_vel(hand_center)
                
                # Display if not headless
                if not self.args.headless:
                    self.draw_visualization(frame, hand_center, cmd_vel)
                    cv2.imshow("Hand Tracking System", frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                else:
                    # Headless status
                    if self.frame_count % 30 == 0:
                        status = "DETECTED" if hand_center else "NOT DETECTED"
                        print(f"[STATUS] Hand: {status}, FPS: {self.fps_counter.get_fps():.1f}")
                
                self.frame_count += 1
                
        except KeyboardInterrupt:
            print("[INFO] Interrupted by user")
        finally:
            self.cleanup()
    
    def draw_visualization(self, frame, hand_center, cmd_vel):
        """Draw enhanced visualization with Cartesian coordinate system."""
        center_x = self.width // 2
        center_y = self.height // 2
        dead_zone = 50
        
        # Draw Cartesian coordinate system
        # Main axes (thicker lines)
        cv2.line(frame, (center_x, 0), (center_x, self.height), (255, 0, 0), 2)  # Y-axis
        cv2.line(frame, (0, center_y), (self.width, center_y), (255, 0, 0), 2)   # X-axis
        
        # Grid lines (thinner)
        for i in range(0, self.width, 80):
            cv2.line(frame, (i, 0), (i, self.height), (100, 100, 100), 1)
        for i in range(0, self.height, 60):
            cv2.line(frame, (0, i), (self.width, i), (100, 100, 100), 1)
            
        # Dead zone circle
        cv2.circle(frame, (center_x, center_y), dead_zone, (0, 255, 0), 2)
        
        # Zone labels (Cartesian coordinate system)
        cv2.putText(frame, "FORWARD (+Y)", (center_x - 60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "BACKWARD (-Y)", (center_x - 70, self.height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "LEFT (-X)", (10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "RIGHT (+X)", (self.width - 100, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "STOP", (center_x - 25, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add quadrant labels
        cv2.putText(frame, "QI", (center_x + 60, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "QII", (center_x - 80, center_y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "QIII", (center_x - 80, center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "QIV", (center_x + 60, center_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        
        # Coordinate system origin marker
        cv2.circle(frame, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.putText(frame, "(0,0)", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add axis arrows to show direction
        # X-axis arrow (right = positive)
        cv2.arrowedLine(frame, (center_x + 20, center_y), (center_x + 40, center_y), (255, 255, 255), 2)
        cv2.putText(frame, "+X", (center_x + 45, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        # Y-axis arrow (up = positive)  
        cv2.arrowedLine(frame, (center_x, center_y - 20), (center_x, center_y - 40), (255, 255, 255), 2)
        cv2.putText(frame, "+Y", (center_x + 5, center_y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw hand center and tracking info
        if hand_center:
            # Hand position marker
            cv2.circle(frame, hand_center, 12, (0, 0, 255), -1)
            cv2.circle(frame, hand_center, 15, (255, 255, 255), 2)
            
            # Calculate relative coordinates from center (Cartesian system)
            rel_x = hand_center[0] - center_x
            rel_y = center_y - hand_center[1]  # Flip Y for proper Cartesian coordinates (positive Y = up)
            
            # Distance from center
            distance = int(np.sqrt(rel_x*rel_x + rel_y*rel_y))
            
            # Coordinate label near hand
            coord_text = f"({rel_x:+d},{rel_y:+d})"
            label_x = hand_center[0] + 20
            label_y = hand_center[1] - 20
            
            # Ensure label stays on screen
            if label_x > self.width - 100:
                label_x = hand_center[0] - 100
            if label_y < 30:
                label_y = hand_center[1] + 30
                
            # Draw coordinate label with background
            cv2.rectangle(frame, (label_x - 5, label_y - 20), (label_x + 95, label_y + 5), (0, 0, 0), -1)
            cv2.putText(frame, coord_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw line from center to hand
            cv2.line(frame, (center_x, center_y), hand_center, (255, 255, 0), 1)
            
        # Status panel (top-left)
        panel_height = 120
        cv2.rectangle(frame, (0, 0), (300, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (300, panel_height), (255, 255, 255), 2)
        
        # Status text
        y_offset = 25
        cv2.putText(frame, f"FPS: {self.fps_counter.get_fps():.1f}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if hand_center:
            y_offset += 25
            cv2.putText(frame, f"Position: ({hand_center[0]}, {hand_center[1]})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Zone: {cmd_vel['zone'].upper()}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Linear: {cmd_vel['linear_x']:+.3f} m/s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            y_offset += 20
            cv2.putText(frame, f"Angular: {cmd_vel['angular_z']:+.3f} rad/s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            y_offset += 25
            cv2.putText(frame, "No hand detected", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def cleanup(self, signum=None, frame=None):
        """Clean up resources."""
        if not self.running:
            return
            
        self.running = False
        print("[INFO] Shutting down...")
        
        if hasattr(self, 'camera'):
            self.camera.release()
            
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = HandTrackingApp()
    app.run_detection_loop()
