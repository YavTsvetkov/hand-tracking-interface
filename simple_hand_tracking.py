#!/usr/bin/env python3
"""
Simplified hand tracking for accurate coordinate extraction.
Removes unnecessary complexity while maintaining accuracy.
"""

import cv2
import numpy as np
import mediapipe as mp
import time

class SimpleHandTracker:
    """Simplified hand tracking focused on accurate coordinate extraction."""
    
    def __init__(self, camera_width=640, camera_height=480):
        self.camera_width = camera_width
        self.camera_height = camera_height
        
        # Initialize MediaPipe Hands (simpler than palm detection)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Simple smoothing
        self.prev_center = None
        self.smoothing_factor = 0.7  # Higher = less smoothing
        
    def process_frame(self, frame):
        """Process frame and return hand center coordinates."""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = self.hands.process(rgb_frame)
        
        hand_center = None
        if results.multi_hand_landmarks:
            # Get first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Calculate center from all landmarks
            x_coords = [landmark.x for landmark in hand_landmarks.landmark]
            y_coords = [landmark.y for landmark in hand_landmarks.landmark]
            
            # Convert normalized coordinates to pixels
            center_x = int(np.mean(x_coords) * self.camera_width)
            center_y = int(np.mean(y_coords) * self.camera_height)
            
            hand_center = (center_x, center_y)
            
            # Simple smoothing
            if self.prev_center is not None:
                smoothed_x = int(self.prev_center[0] + self.smoothing_factor * (center_x - self.prev_center[0]))
                smoothed_y = int(self.prev_center[1] + self.smoothing_factor * (center_y - self.prev_center[1]))
                hand_center = (smoothed_x, smoothed_y)
            
            self.prev_center = hand_center
            
            # Draw landmarks for visualization
            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
        return hand_center, frame

class SimpleCartesianController:
    """Simple 4-quadrant Cartesian controller for cmd_vel."""
    
    def __init__(self, frame_width=640, frame_height=480, dead_zone=50):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.center_x = frame_width // 2
        self.center_y = frame_height // 2
        self.dead_zone = dead_zone
        
        # Maximum speeds
        self.max_linear_speed = 0.5  # m/s
        self.max_angular_speed = 1.0  # rad/s
        
    def get_cmd_vel(self, hand_position):
        """Convert hand position to cmd_vel commands."""
        if hand_position is None:
            return {'linear_x': 0.0, 'angular_z': 0.0, 'zone': 'none'}
        
        x, y = hand_position
        
        # Calculate offset from center
        dx = x - self.center_x
        dy = y - self.center_y
        distance = np.sqrt(dx*dx + dy*dy)
        
        # Check dead zone
        if distance < self.dead_zone:
            return {'linear_x': 0.0, 'angular_z': 0.0, 'zone': 'center'}
        
        # Determine quadrant and calculate velocities
        # Coordinate system: 
        # - Y: negative = forward, positive = backward
        # - X: negative = left turn, positive = right turn
        
        # Normalize to [-1, 1] range
        linear_factor = -dy / (self.frame_height / 2)  # Flip Y axis
        angular_factor = -dx / (self.frame_width / 2)  # Flip X axis for left/right
        
        # Clamp to [-1, 1]
        linear_factor = np.clip(linear_factor, -1.0, 1.0)
        angular_factor = np.clip(angular_factor, -1.0, 1.0)
        
        # Apply distance-based scaling (closer to center = less movement)
        max_distance = min(self.frame_width, self.frame_height) / 2
        intensity = min(distance / max_distance, 1.0)
        
        # Calculate final velocities
        linear_x = linear_factor * intensity * self.max_linear_speed
        angular_z = angular_factor * intensity * self.max_angular_speed
        
        # Determine zone
        zone = self._get_zone(dx, dy)
        
        return {
            'linear_x': linear_x,
            'angular_z': angular_z,
            'zone': zone,
            'intensity': intensity
        }
    
    def _get_zone(self, dx, dy):
        """Determine which quadrant/zone the hand is in."""
        if abs(dx) > abs(dy):
            return 'left' if dx < 0 else 'right'
        else:
            return 'forward' if dy < 0 else 'backward'

def main():
    """Simple test application."""
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize components
    tracker = SimpleHandTracker(640, 480)
    controller = SimpleCartesianController(640, 480, dead_zone=50)
    
    print("Simple Hand Tracking Started. Press 'q' to quit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            hand_center, display_frame = tracker.process_frame(frame)
            
            # Get cmd_vel
            cmd_vel = controller.get_cmd_vel(hand_center)
            
            # Draw center cross and dead zone with enhanced Cartesian system
            center_x, center_y = 320, 240
            
            # Draw Cartesian coordinate system
            # Main axes (thicker lines)
            cv2.line(display_frame, (center_x, 0), (center_x, 480), (255, 0, 0), 2)  # Y-axis
            cv2.line(display_frame, (0, center_y), (640, center_y), (255, 0, 0), 2)   # X-axis
            
            # Grid lines (thinner)
            for i in range(0, 640, 80):
                cv2.line(display_frame, (i, 0), (i, 480), (100, 100, 100), 1)
            for i in range(0, 480, 60):
                cv2.line(display_frame, (0, i), (640, i), (100, 100, 100), 1)
                
            # Dead zone circle
            cv2.circle(display_frame, (center_x, center_y), 50, (0, 255, 0), 2)
            
            # Zone labels
            cv2.putText(display_frame, "FORWARD", (center_x - 40, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "BACKWARD", (center_x - 50, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "LEFT", (10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "RIGHT", (570, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, "STOP", (center_x - 25, center_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Coordinate system origin marker
            cv2.circle(display_frame, (center_x, center_y), 5, (255, 255, 255), -1)
            cv2.putText(display_frame, "(0,0)", (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw hand center with enhanced visualization
            if hand_center:
                # Hand position marker
                cv2.circle(display_frame, hand_center, 12, (0, 0, 255), -1)
                cv2.circle(display_frame, hand_center, 15, (255, 255, 255), 2)
                
                # Calculate relative coordinates from center
                rel_x = hand_center[0] - 320
                rel_y = hand_center[1] - 240
                
                # Distance from center
                distance = int(np.sqrt(rel_x*rel_x + rel_y*rel_y))
                
                # Coordinate label near hand
                coord_text = f"({rel_x:+d},{rel_y:+d})"
                label_x = hand_center[0] + 20
                label_y = hand_center[1] - 20
                
                # Ensure label stays on screen
                if label_x > 540:
                    label_x = hand_center[0] - 100
                if label_y < 30:
                    label_y = hand_center[1] + 30
                    
                # Draw coordinate label with background
                cv2.rectangle(display_frame, (label_x - 5, label_y - 20), (label_x + 95, label_y + 5), (0, 0, 0), -1)
                cv2.putText(display_frame, coord_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Draw line from center to hand
                cv2.line(display_frame, (320, 240), hand_center, (255, 255, 0), 1)
                
                # Status panel
                cv2.rectangle(display_frame, (0, 0), (300, 120), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (0, 0), (300, 120), (255, 255, 255), 2)
                
                # Enhanced status info
                cv2.putText(display_frame, f"Position: ({hand_center[0]}, {hand_center[1]})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Relative: ({rel_x:+d}, {rel_y:+d})", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Distance: {distance}px", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(display_frame, f"Zone: {cmd_vel['zone'].upper()}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                cv2.putText(display_frame, f"L:{cmd_vel['linear_x']:+.2f} A:{cmd_vel['angular_z']:+.2f}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                print(f"Position: {hand_center}, Zone: {cmd_vel['zone']}, Linear: {cmd_vel['linear_x']:.2f}, Angular: {cmd_vel['angular_z']:.2f}")
            
            # Display frame
            cv2.imshow('Simple Hand Tracking', display_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
