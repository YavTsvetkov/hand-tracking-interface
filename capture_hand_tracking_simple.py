#!/usr/bin/env python3
"""
Simplified Hand Tracking Script for Raspberry Pi 5
Streamlined version with only essential functionality
"""

import cv2
import numpy as np
import time
import argparse
import sys
import queue
import threading
import subprocess
import tempfile
import os

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# Import custom hand detection helpers
try:
    from hand_detection_helper import HandTracker, PositionValidator
except ImportError:
    print("[WARNING] Could not import hand_detection_helper.py")
    # Define fallback minimal implementations if module not available
    class HandTracker:
        def __init__(self): self.is_hand_present = False
        def update(self, position=None, confidence=None): return position is not None
        def get_tracking_quality(self): return 0.5
        def is_tracking_stable(self): return True
        def reset(self): pass
        
    class PositionValidator:
        def __init__(self, max_speed=800, min_speed=0.5, width=640, height=480): pass
        def is_valid(self, position, current_time=None): return position is not None
        def is_suspiciously_still(self, threshold=15): return False
        def reset(self): pass

def parse_args():
    """Parse command line arguments - only essential flags"""
    parser = argparse.ArgumentParser(description="Simplified hand tracking on Raspberry Pi 5")
    parser.add_argument('--model', type=str, default='hand_landmark_lite.tflite', 
                       help='Path to TFLite hand landmark model')
    parser.add_argument('--res', type=str, required=True, 
                       help='Camera resolution WxH (e.g., 640x480, 320x240)')
    parser.add_argument('--fps', type=int, default=30,
                       help='Camera framerate (default: 30)')
    parser.add_argument('--frame_skip', type=int, default=1, 
                       help='Process every Nth frame (default: 1 for max accuracy)')
    parser.add_argument('--headless', action='store_true', 
                       help='Run without GUI display')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    parser.add_argument('--confidence', type=float, default=0.6,
                       help='Minimum confidence threshold (0-1, default: 0.6)')
    parser.add_argument('--smoothing', type=float, default=0.4,
                       help='Coordinate smoothing factor (0-1, default: 0.4)')
    parser.add_argument('--crop_factor', type=float, default=0.8,
                       help='Center crop factor for better accuracy (0-1, default: 0.8)')
    parser.add_argument('--max_jump', type=int, default=150,
                       help='Maximum pixel jump to filter noise (default: 150)')
    parser.add_argument('--detection_loss_frames', type=int, default=5,
                       help='Number of frames before declaring hand lost (default: 5)')
    parser.add_argument('--stability_threshold', type=float, default=0.6,
                       help='Minimum tracking quality for stable detection (0-1, default: 0.6)')
    parser.add_argument('--false_positive_threshold', type=int, default=5,
                       help='Frames of suspicious detection before rejection (default: 5)')
    return parser.parse_args()

class LibcameraCapture:
    """Simplified libcamera capture for hand tracking"""
    
    def __init__(self, width, height, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3 // 2  # YUV420
        
        # Create FIFO
        self.temp_dir = tempfile.mkdtemp()
        self.fifo_path = os.path.join(self.temp_dir, 'camera_fifo')
        os.mkfifo(self.fifo_path)
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # Start libcamera process with optimizations
        self.process = subprocess.Popen([
            'libcamera-vid', 
            '--width', str(width), 
            '--height', str(height),
            '--framerate', str(fps), 
            '--timeout', '0', 
            '--codec', 'yuv420',
            '--flush',  # Reduce latency
            '--inline',  # Inline headers for reduced latency
            '--listen',  # No preview window
            '--nopreview',  # Explicitly disable any preview
            '--buffer-count', '2',  # Minimal buffer count
            '--output', self.fifo_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)  # Allow camera more time to initialize
        self.fifo = open(self.fifo_path, 'rb')
        
        # Start frame reader thread
        self.reader_thread = threading.Thread(target=self._frame_reader)
        self.reader_thread.daemon = True
        self.reader_thread.start()
    
    def _frame_reader(self):
        """Background frame reading"""
        while self.running:
            try:
                # Read frame data
                data = self.fifo.read(self.frame_size)
                if not data or len(data) != self.frame_size:
                    # If we didn't get a complete frame, try again
                    if self.running:
                        time.sleep(0.001)
                    continue
                
                # Convert YUV420 to BGR
                yuv = np.frombuffer(data, dtype=np.uint8)
                y_size = self.width * self.height
                uv_size = y_size // 4
                
                y_plane = yuv[:y_size].reshape(self.height, self.width)
                u_plane = yuv[y_size:y_size + uv_size].reshape(self.height // 2, self.width // 2)
                v_plane = yuv[y_size + uv_size:y_size + 2 * uv_size].reshape(self.height // 2, self.width // 2)
                
                yuv420 = np.zeros((self.height * 3 // 2, self.width), dtype=np.uint8)
                yuv420[:self.height, :] = y_plane
                
                uv_interleaved = np.empty((self.height // 2, self.width), dtype=np.uint8)
                uv_interleaved[:, 0::2] = u_plane
                uv_interleaved[:, 1::2] = v_plane
                yuv420[self.height:, :] = uv_interleaved
                
                bgr_frame = cv2.cvtColor(yuv420, cv2.COLOR_YUV2BGR_NV12)
                
                # Drop old frames, keep only latest for real-time performance
                try:
                    # Clear queue if full to maintain real-time performance
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.frame_queue.put(bgr_frame, block=False)
                except queue.Full:
                    pass  # Skip frame if still full
                    
            except Exception as e:
                print(f"[DEBUG] Frame reading error: {str(e)}")
                if self.running:
                    time.sleep(0.001)  # Short sleep on error
                continue  # Continue the loop instead of breaking
    
    def read(self):
        """Get latest frame with minimal timeout for real-time performance"""
        try:
            frame = self.frame_queue.get(timeout=0.5)  # Increased timeout for better reliability
            if frame is not None:
                return True, frame
            return False, None
        except queue.Empty:
            # If queue is empty, wait a bit and return no frame
            return False, None
    
    def release(self):
        """Clean up resources"""
        self.running = False
        try:
            print("[INFO] Shutting down camera...")
            if hasattr(self, 'reader_thread') and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=2.0)
            if hasattr(self, 'fifo'):
                self.fifo.close()
            if hasattr(self, 'process'):
                self.process.terminate()
                self.process.wait(timeout=5)
            if hasattr(self, 'fifo_path') and os.path.exists(self.fifo_path):
                os.unlink(self.fifo_path)
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
            print("[INFO] Camera resources cleaned up")
        except Exception as e:
            print(f"[ERROR] Error during cleanup: {e}")
            # Force cleanup if needed
            if hasattr(self, 'process'):
                try:
                    self.process.kill()
                except:
                    pass

# Step 1: Model and Inference Optimizations
def load_model(model_path):
    """Load TFLite model with optimizations"""
    try:
        # Basic interpreter with multi-threading for Pi 5
        interpreter = Interpreter(model_path=model_path, num_threads=4)
    except Exception as e:
        print(f"[WARNING] Failed to load model with optimizations: {e}")
        interpreter = Interpreter(model_path=model_path)
    
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, frame_rgb, crop_factor=0.8):
    """Run hand tracking inference with optimizations"""
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    
    # Optimized input preparation with ROI cropping for better accuracy
    h, w = frame_rgb.shape[:2]
    input_size = (input_shape[2], input_shape[1])
    
    # Optional: Crop center region for better hand detection (adjust as needed)
    crop_h, crop_w = int(h * crop_factor), int(w * crop_factor)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    cropped_frame = frame_rgb[start_h:start_h + crop_h, start_w:start_w + crop_w]
    
    # Resize with optimized interpolation
    input_tensor = cv2.resize(cropped_frame, input_size, interpolation=cv2.INTER_LINEAR)
    
    # Normalize efficiently
    input_tensor = input_tensor.astype(np.float32) * (1.0 / 255.0)
    input_tensor = np.expand_dims(input_tensor, axis=0)
    
    # Run inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    inference_time = (time.time() - start) * 1000
    
    # Get output
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output, inference_time, (start_h, start_w, crop_factor)

# Step 2: Enhanced Coordinate Processing with Smoothing
class CoordinateSmoothing:
    """Kalman-like smoothing for palm coordinates"""
    
    def __init__(self, alpha=0.3):
        self.alpha = alpha  # Smoothing factor (0-1, lower = more smoothing)
        self.prev_x = None
        self.prev_y = None
        self.velocity_x = 0
        self.velocity_y = 0
        
    def smooth(self, x, y):
        """Apply exponential smoothing with velocity prediction"""
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y
            return x, y
        
        # Update velocity
        self.velocity_x = x - self.prev_x
        self.velocity_y = y - self.prev_y
        
        # Apply smoothing
        smoothed_x = self.alpha * x + (1 - self.alpha) * (self.prev_x + self.velocity_x * 0.5)
        smoothed_y = self.alpha * y + (1 - self.alpha) * (self.prev_y + self.velocity_y * 0.5)
        
        self.prev_x, self.prev_y = smoothed_x, smoothed_y
        return int(smoothed_x), int(smoothed_y)
    
    def predict_next(self):
        """Predict next position based on velocity"""
        if self.prev_x is None:
            return None, None
        return int(self.prev_x + self.velocity_x), int(self.prev_y + self.velocity_y)
    
    def reset(self):
        """Reset tracking state when detection is lost"""
        self.prev_x = None
        self.prev_y = None
        self.velocity_x = 0
        self.velocity_y = 0

def draw_wrist(frame, landmarks, width, height, crop_info=None, show_coords=True):
    """Draw wrist landmark with coordinates and crop compensation"""
    if len(landmarks) == 0:
        return None
    
    # Get wrist coordinates (first landmark)
    x, y = landmarks[0][0], landmarks[0][1]
    
    # Normalize if needed
    if x > 1.0:
        x = x / width
    if y > 1.0:
        y = y / height
    
    # Compensate for cropping if used
    if crop_info:
        start_h, start_w, crop_factor = crop_info
        # Adjust coordinates back to original frame
        x = (x * crop_factor) + (start_w / width)
        y = (y * crop_factor) + (start_h / height)
    
    cx, cy = int(x * width), int(y * height)
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    
    # Draw wrist with enhanced visibility
    cv2.circle(frame, (cx, cy), 18, (0, 0, 255), -1)  # Larger red circle
    cv2.circle(frame, (cx, cy), 22, (255, 255, 255), 3)  # White border
    cv2.circle(frame, (cx, cy), 8, (255, 255, 0), -1)   # Yellow center dot
    
    # Show coordinates
    if show_coords:
        coord_text = f"Palm: ({cx}, {cy})"
        cv2.putText(frame, coord_text, (cx + 30, cy - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, coord_text, (cx + 30, cy - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)  # Black outline
    
    return (cx, cy)

# Step 3: Add Confidence-Based Filtering
def extract_landmarks_with_confidence(output, confidence_threshold=0.5):
    """Extract landmarks with confidence filtering"""
    try:
        # Handle different output formats safely
        if len(output.shape) == 3:
            landmarks = output[0].reshape(-1, 3)
        elif len(output.shape) == 2:
            landmarks = output.reshape(-1, 3)
        else:
            # Unexpected shape
            return np.array([]), False
        
        # Early return for empty results
        if landmarks.shape[0] == 0:
            return np.array([]), False
        
        # Check if the model actually detected a hand
        # MediaPipe hand models often have a presence score in first landmark
        presence_score = landmarks[0, 2] if landmarks.shape[0] > 0 and landmarks.shape[1] > 2 else 0
        
        # Some models might use the wrist confidence as hand presence score
        wrist_confidence_threshold = confidence_threshold * 1.2  # Higher threshold for wrist
        if presence_score < wrist_confidence_threshold:
            # No hand detected with sufficient confidence
            return np.array([]), False
        
        # For MediaPipe models, the 3rd dimension often represents confidence/visibility
        if landmarks.shape[1] >= 3:
            # Check if third dimension looks like confidence (0-1 range)
            confidences = landmarks[:, 2]
            if np.all(confidences <= 1.0) and np.all(confidences >= 0.0):
                # Calculate overall confidence for the detection (with higher weight to wrist point)
                critical_points = [0, 5, 9, 13, 17]  # Wrist and finger base points
                critical_confidence = np.mean(confidences[critical_points]) if len(confidences) > 17 else confidences[0]
                avg_confidence = np.mean(confidences)
                
                # Combined confidence score
                detection_confidence = critical_confidence * 0.7 + avg_confidence * 0.3
                high_confidence = detection_confidence >= confidence_threshold
                
                # Filter landmarks to keep only reliable ones
                # Only keep points with good confidence
                high_conf_mask = confidences >= confidence_threshold * 0.8
                
                # Always keep wrist point (index 0) if it has reasonable confidence
                if confidences[0] >= confidence_threshold * 0.7:
                    high_conf_mask[0] = True
                
                if np.any(high_conf_mask) and high_confidence:
                    # Return with confidence status
                    return landmarks[high_conf_mask], True
                else:
                    # Add debug print if needed
                    # print(f"Confidence too low: avg={avg_confidence:.2f}, critical={critical_confidence:.2f}")
                    return np.array([]), False  # Return empty array for low confidence
        
        # If we can't determine confidence, assume it's not reliable
        return np.array([]), False
    
    except Exception as e:
        print(f"[ERROR] Landmark extraction error: {e}")
        return np.array([]), False

def validate_palm_position(cx, cy, width, height, prev_pos=None, max_jump=150):
    """Validate palm position to filter out noise and false positives"""
    # Check bounds
    if cx < 0 or cx >= width or cy < 0 or cy >= height:
        return False
    
    # Check for unrealistic jumps (sudden large movements)
    if prev_pos is not None:
        prev_cx, prev_cy = prev_pos
        distance = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
        
        # Detect implausible jump in position
        if distance > max_jump:
            return False
        
        # Additional check for false positive at fixed location
        # If the position is exactly the same for multiple frames, it might be a false positive
        if distance < 0.1:  # Almost no movement at all
            # This could be a false positive fixed position
            # We'll return True but this should increment a counter elsewhere
            return True
    
    # Check for common false positive areas (like edges/corners)
    # Raspberry Pi cameras sometimes falsely detect at edges
    margin = width * 0.05  # 5% margin
    
    # Check if near edges - these are often false detection areas
    if (cx < margin or cx > width - margin or cy < margin or cy > height - margin):
        # Position is at edge, be more skeptical
        # Not rejecting outright, but should be treated with higher scrutiny
        return True
    
    # Check for fixed positions that are known false positives
    # If seeing detection consistently at exact same pixel, that's suspicious
    
    return True

def main():
    args = parse_args()
    
    # Parse resolution
    width, height = map(int, args.res.split('x'))
    
    # Initialize camera - only libcamera, no fallbacks
    print("[INFO] Initializing libcamera...")
    
    # Check if libcamera-vid is available
    try:
        subprocess.run(['libcamera-vid', '--help'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1)
        print("[INFO] libcamera-vid found")
    except (subprocess.SubprocessError, FileNotFoundError) as e:
        print(f"[ERROR] libcamera-vid not available: {e}")
        print("[ERROR] Please ensure libcamera-apps is installed")
        sys.exit(1)
    
    try:
        cap = LibcameraCapture(width, height, args.fps)
    except Exception as e:
        print(f"[ERROR] Failed to initialize camera: {e}")
        sys.exit(1)
    
    # Check camera is providing frames
    print("[INFO] Checking camera frame capture...")
    start_time = time.time()
    while time.time() - start_time < 5:  # Try for 5 seconds
        ret, test_frame = cap.read()
        if ret and test_frame is not None:
            print("[INFO] Camera initialized successfully")
            break
        time.sleep(0.1)
    else:
        print("[ERROR] Camera not providing frames, check connections and permissions")
        cap.release()
        sys.exit(1)
        
    # Load model
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        sys.exit(1)
    
    print(f"[INFO] Loading model: {args.model}")
    interpreter = load_model(args.model)
    
    # Step 4: Update Main Loop Variables and Settings
    # Initialize variables
    prev_time = time.time()
    frame_count = 0
    skip_counter = 0
    last_wrist_pos = None
    last_inf_time = 0
    
    # Detection state tracking
    detection_loss_counter = 0
    max_detection_loss = args.detection_loss_frames  # Use command line argument
    hand_present = False
    
    # Initialize coordinate smoothing
    coord_smoother = CoordinateSmoothing(alpha=args.smoothing)
    confidence_threshold = args.confidence
    
    # Initialize advanced hand tracking
    hand_tracker = HandTracker(history_size=15, 
                              stable_threshold=int(args.detection_loss_frames * 0.6),
                              fixed_pos_threshold=1.5, 
                              max_still_frames=20)
    position_validator = PositionValidator(
        max_speed=800,  # Max pixels per second
        min_speed=0.5,  # Min pixels per second for real motion
        width=width,
        height=height
    )
    
    # Detection quality tracking
    detection_quality = 0.0
    false_positive_counter = 0
    
    print(f"[INFO] Starting hand tracking ({'headless' if args.headless else 'GUI'} mode)")
    print(f"[INFO] Resolution: {width}x{height}, FPS: {args.fps}, Frame skip: {args.frame_skip}")
    print(f"[INFO] Confidence threshold: {confidence_threshold}")
    print(f"[INFO] Smoothing factor: {args.smoothing}")
    print(f"[INFO] Crop factor: {args.crop_factor}")
    print(f"[INFO] Advanced hand tracking enabled")
    
    if not args.headless:
        print("[INFO] Press ESC to exit")
    else:
        print("[INFO] Press Ctrl+C to exit")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            
            frame_count += 1
            
            # Frame skipping logic
            if skip_counter < args.frame_skip:
                skip_counter += 1
                
                # Show last known wrist position for skipped frames only if hand is present
                if last_wrist_pos and hand_present:
                    cv2.circle(frame, last_wrist_pos, 15, (0, 0, 255), -1)
                    cv2.circle(frame, last_wrist_pos, 18, (255, 255, 255), 3)
                    
                    # Show coordinates for skipped frames too
                    coord_text = f"Palm: {last_wrist_pos}"
                    cv2.putText(frame, coord_text, (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                elif not hand_present:
                    # Show no hand detected message
                    cv2.putText(frame, "NO HAND DETECTED", (width//2-140, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Calculate FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time
                
                if not args.headless:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(frame, f"Inference: {last_inf_time:.1f} ms (cached)", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('Hand Tracking - Simplified', frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC
                        break
                continue
            
            # Process frame
            skip_counter = 0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, inf_time, crop_info = run_inference(interpreter, frame_rgb, args.crop_factor)
            last_inf_time = inf_time
            
            # Extract landmarks with confidence filtering
            landmarks, high_confidence = extract_landmarks_with_confidence(output, confidence_threshold)
            
            # Store confidence value for tracking
            current_confidence = landmarks[0, 2] if landmarks.shape[0] >= 1 and landmarks.shape[1] >= 3 else 0
            
            # Check for valid hand detection
            if landmarks.shape[0] >= 1 and high_confidence:
                # Get raw coordinates
                raw_wrist_pos = draw_wrist(frame, landmarks[:1], width, height, crop_info, show_coords=False)
                
                # Update advanced hand tracker
                current_time = time.time()
                if raw_wrist_pos:
                    # Update position validator
                    position_valid = position_validator.is_valid(raw_wrist_pos, current_time)
                    
                    # Update hand tracker only if position seems valid
                    if position_valid:
                        tracker_updated = hand_tracker.update(raw_wrist_pos, current_confidence)
                        # Only reset loss counter if both detection and position are valid
                        detection_loss_counter = 0
                        hand_present = hand_tracker.is_hand_present
                    else:
                        # Position is invalid, count as a detection loss
                        detection_loss_counter += 1
                        tracker_updated = False
                else:
                    # No position detected despite high confidence
                    detection_loss_counter += 1
                    tracker_updated = False
                
                # Display confidence if in debug mode
                if args.debug and landmarks.shape[1] >= 3:
                    conf_value = current_confidence
                    # Color based on tracking quality
                    tracking_quality = hand_tracker.get_tracking_quality()
                    if tracking_quality > 0.7:
                        conf_color = (0, 255, 0)  # Good (green)
                    elif tracking_quality > 0.4:
                        conf_color = (0, 255, 255)  # Medium (yellow)
                    else:
                        conf_color = (0, 0, 255)  # Poor (red)
                        
                    cv2.putText(frame, f"Conf: {conf_value:.2f}", (width - 200, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
                    cv2.putText(frame, f"Track: {tracking_quality:.2f}", (width - 200, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, conf_color, 2)
                
                # Validate position and check for false positives
                if raw_wrist_pos and validate_palm_position(raw_wrist_pos[0], raw_wrist_pos[1], width, height, last_wrist_pos, args.max_jump):
                    # Check for suspiciously still positions (possible false positive)
                    is_suspicious_still = position_validator.is_suspiciously_still(threshold=15)
                    tracking_stable = hand_tracker.is_tracking_stable()
                    
                    # Apply smoothing only if tracking is stable
                    if tracking_stable and not is_suspicious_still:
                        # Good detection - apply smoothing
                        smoothed_x, smoothed_y = coord_smoother.smooth(raw_wrist_pos[0], raw_wrist_pos[1])
                        wrist_pos = (smoothed_x, smoothed_y)
                        last_wrist_pos = wrist_pos
                        
                        # Draw final smoothed position with high confidence (green)
                        cv2.circle(frame, wrist_pos, 18, (0, 255, 0), -1)  # Green for smoothed
                        cv2.circle(frame, wrist_pos, 22, (255, 255, 255), 3)  # White border
                        cv2.circle(frame, wrist_pos, 8, (255, 255, 0), -1)   # Yellow center
                        
                        # Show tracking quality indicator
                        quality = hand_tracker.get_tracking_quality()
                        quality_text = f"Quality: {quality:.2f}"
                        cv2.putText(frame, quality_text, (10, 210), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
                        # Show both raw and smoothed coordinates
                        coord_text = f"Smooth: {wrist_pos}"
                        cv2.putText(frame, coord_text, (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Reset false positive counter
                        false_positive_counter = 0
                    else:
                        # Suspicious detection - might be a false positive
                        # Use last known good position but mark as uncertain
                        wrist_pos = raw_wrist_pos  # Use unsmoothed position for visualization
                        
                        # Draw with "uncertain" color (orange)
                        cv2.circle(frame, wrist_pos, 18, (0, 165, 255), -1)  # Orange
                        cv2.circle(frame, wrist_pos, 22, (255, 255, 255), 2)  # Thinner white border
                        
                        # Show warning
                        cv2.putText(frame, "UNCERTAIN DETECTION", (10, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        
                        # Increment false positive counter
                        false_positive_counter += 1
                        
                        # If too many suspicious frames, consider it a lost detection
                        if false_positive_counter > args.false_positive_threshold:
                            detection_loss_counter += 1
                    
                    # Show debug info if requested
                    if args.debug:
                        raw_text = f"Raw: {raw_wrist_pos}"
                        cv2.putText(frame, raw_text, (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                        print(f"Frame {frame_count}: Wrist Raw={raw_wrist_pos}, Smooth={wrist_pos}, " + 
                              f"Conf: {current_confidence:.2f}, Quality: {hand_tracker.get_tracking_quality():.2f}, " +
                              f"Inference: {inf_time:.1f}ms")
                else:
                    # Invalid position detected, increment loss counter
                    detection_loss_counter += 1
                    
                    # Use prediction if validation fails but only if we haven't lost detection for too long
                    if detection_loss_counter < max_detection_loss and hand_tracker.get_tracking_quality() > 0.3:
                        # Try smoothing prediction first
                        pred_x, pred_y = coord_smoother.predict_next()
                        if pred_x is not None:
                            wrist_pos = (pred_x, pred_y)
                            cv2.circle(frame, wrist_pos, 15, (0, 255, 255), -1)  # Cyan for prediction
                            cv2.putText(frame, "PREDICTED", (10, 180), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        # Mark tracking as potentially lost
                        false_positive_counter += 1
            else:
                # No hand detected by the model
                # Update hand tracker with no position
                hand_tracker.update(None, None)
                detection_loss_counter += 1
                
                # If we've lost detection for too many frames, reset tracking
                if detection_loss_counter >= max_detection_loss:
                    if hand_present:  # Only log when state changes
                        if args.debug:
                            print(f"Frame {frame_count}: Hand detection lost")
                        hand_present = False
                        # Reset tracking states
                        coord_smoother.reset()
                        position_validator.reset()
                        hand_tracker.reset()
                        last_wrist_pos = None
                        false_positive_counter = 0
                    
                    # Draw NO HAND DETECTED indicator
                    cv2.putText(frame, "NO HAND DETECTED", (width//2-140, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Show quality indicator
                    quality = hand_tracker.get_tracking_quality()
                    cv2.putText(frame, f"Detection quality: {quality:.2f}", (10, 210), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # Brief loss of detection, show last known position with visual indication
                    # Check if we still have tracking quality
                    tracking_quality = hand_tracker.get_tracking_quality()
                    
                    if last_wrist_pos and tracking_quality > 0.2:
                        # We still have some tracking quality, show last position as uncertain
                        cv2.circle(frame, last_wrist_pos, 15, (0, 165, 255), -1)  # Orange for uncertain
                        cv2.putText(frame, "TRACKING UNCERTAIN", (10, 180), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        cv2.putText(frame, f"Quality: {tracking_quality:.2f}", (10, 210), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        # No reliable tracking, show searching indicator
                        cv2.putText(frame, "SEARCHING FOR HAND...", (width//2-160, height//2), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                
                if args.debug and detection_loss_counter == max_detection_loss:
                    print(f"Frame {frame_count}: No hand detected, confidence too low or false positive detected")
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            if not args.headless:
                # Display info
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Inference: {inf_time:.1f} ms", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
                # Get tracking quality for status
                tracking_quality = hand_tracker.get_tracking_quality()
                
                # Display hand detection status with tracking quality
                if hand_present and tracking_quality > 0.7:
                    status_text = "DETECTED ✓"
                    status_color = (0, 255, 0)  # Green for good detection
                elif hand_present:
                    status_text = "UNCERTAIN"
                    status_color = (0, 255, 255)  # Yellow for uncertain
                else:
                    status_text = "NO HAND"
                    status_color = (0, 0, 255)  # Red for no hand
                
                cv2.putText(frame, f"Status: {status_text}", (width - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Add false positive indicator if needed
                if false_positive_counter > 0:
                    fp_text = f"False Detection?: {false_positive_counter}"
                    cv2.putText(frame, fp_text, (width - 200, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow('Hand Tracking - Simplified', frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break
            else:
                # Print coordinates in headless mode
                if last_wrist_pos and frame_count % 30 == 0:  # Print every 30 frames
                    print(f"Frame {frame_count}: Palm at {last_wrist_pos}, FPS: {fps:.1f}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
    finally:
        print("[INFO] Cleaning up...")
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Done")

if __name__ == '__main__':
    main()
