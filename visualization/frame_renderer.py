"""
Frame rendering coordinator for visualization components.
"""

import cv2
import numpy as np
import time


class FrameRenderer:
    """Coordinates all visualization components for rendering frames."""
    
    def __init__(self, width=640, height=480, debug_mode=False):
        """Initialize frame renderer with frame dimensions."""
        self.width = width
        self.height = height
        self.debug_mode = debug_mode
        
        # Last warning/error time for temporary message display
        self.last_warning_time = 0
        self.last_error_time = 0
        self.message_duration = 2.0  # seconds
        
        # Current warning/error messages
        self.warning_message = None
        self.error_message = None
    
    def render_frame(self, frame, landmarks=None, wrist_position=None,
                    tracking_quality=0.0, is_hand_present=False, 
                    is_valid=True, fps=0.0, cmd_vel_values=None, 
                    renderer=None, status_display=None):
        """Render visualization on frame using specified components."""
        # Skip rendering if frame is None
        if frame is None:
            return None
            
        # Make a copy to avoid modifying the original frame
        output_frame = frame.copy()
        # Draw coordinate axes (green)
        self._draw_axes(output_frame)
        
        # Use renderer to draw landmarks if available
        if renderer is not None and landmarks is not None:
            renderer.draw_landmarks(output_frame, landmarks, tracking_quality)
            
            if wrist_position is not None:
                renderer.draw_wrist_position(output_frame, wrist_position, 
                                          tracking_quality, is_valid)
         # Use status display to show tracking status if available
        if status_display is not None:
            status_display.draw_status_box(output_frame, is_hand_present,
                                         tracking_quality, wrist_position, is_valid, fps)
            
            # Draw cmd_vel display if available
            if cmd_vel_values is not None:
                status_display.draw_cmd_vel_display(output_frame, cmd_vel_values)
            
            # Display any active warnings or errors
            current_time = time.time()
            
            # Show warning if it's still within duration
            if (self.warning_message is not None and 
                current_time - self.last_warning_time < self.message_duration):
                status_display.draw_warning(output_frame, self.warning_message)
            else:
                self.warning_message = None
                
            # Show error if it's still within duration
            if (self.error_message is not None and 
                current_time - self.last_error_time < self.message_duration):
                status_display.draw_error(output_frame, self.error_message)
            else:
                self.error_message = None
                
        # Add debug information if enabled
        if self.debug_mode:
            self._add_debug_info(output_frame, fps, tracking_quality, is_hand_present, is_valid)
            
        return output_frame
    
    def show_warning(self, message):
        """Show a temporary warning message."""
        self.warning_message = message
        self.last_warning_time = time.time()
    
    def show_error(self, message):
        """Show a temporary error message."""
        self.error_message = message
        self.last_error_time = time.time()
    
    def _add_debug_info(self, frame, fps, tracking_quality, is_hand_present, is_valid):
        """Add debug information overlay."""
        # Debug background
        cv2.rectangle(frame, (10, frame.shape[0] - 80), 
                     (200, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, frame.shape[0] - 80), 
                     (200, frame.shape[0] - 10), (255, 255, 255), 1)
        
        # Debug info
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"DEBUG MODE", (20, frame.shape[0] - 60), 
                   font, 0.5, (255, 255, 0), 1)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 40), 
                   font, 0.5, (255, 255, 255), 1)
        
        # Track state
        state = "DETECTED" if is_hand_present else "NO HAND"
        if is_hand_present and not is_valid:
            state = "FALSE POSITIVE"
            
        cv2.putText(frame, f"State: {state}", (20, frame.shape[0] - 20), 
                   font, 0.5, (255, 255, 255), 1)
    
    def _draw_axes(self, frame, color=(0, 255, 0), thickness=2):
        """Draw simple X–Y axes crossing at frame center."""
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        length = min(w, h) // 2  # extend from center to edges
        # Horizontal axis
        cv2.line(frame, (cx - length, cy), (cx + length, cy), color, thickness)
        # Vertical axis
        cv2.line(frame, (cx, cy - length), (cx, cy + length), color, thickness)