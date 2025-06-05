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
    from tensorflow.lite import Interpreter

# ---------------------- Argument Parsing ----------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Real-time hand tracking on Raspberry Pi 5 with Pi Camera and TFLite model.")
    parser.add_argument('--model', type=str, default='hand_landmark_lite.tflite', help='Path to TFLite hand landmark model.')
    parser.add_argument('--res', type=str, default='640x480', help='Camera resolution WxH, e.g., 640x480, 800x600 or 320x240.')
    parser.add_argument('--backend', type=str, default='auto', choices=['libcamera', 'v4l2', 'auto'], help='Video backend to use.')
    parser.add_argument('--show_coords', action='store_true', help='Show coordinate text on palm center.')
    parser.add_argument('--frame_skip', type=int, default=2, help='Process every Nth frame (default: 2, meaning process every 3rd frame).')
    parser.add_argument('--test_camera', action='store_true', help='Test camera access methods and exit.')
    parser.add_argument('--headless', action='store_true', help='Run without GUI display (for testing).')
    parser.add_argument('--debug', action='store_true', help='Enable debug output.')
    parser.add_argument('--troubleshoot', action='store_true', help='Run comprehensive camera troubleshooting.')
    return parser.parse_args()

# ---------------------- Video Capture Setup ----------------------
class LibcameraCapture:
    """Optimized Libcamera-based capture using subprocess and FIFO
    
    This implementation uses libcamera-vid with --nopreview to capture frames
    directly to a FIFO buffer without any preview window, maximizing performance
    for real-time hand tracking. Only the processed OpenCV window is displayed.
    """
    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3 // 2  # YUV420 format
        
        # Create temporary directory and FIFO
        self.temp_dir = tempfile.mkdtemp()
        self.fifo_path = os.path.join(self.temp_dir, 'camera_fifo')
        os.mkfifo(self.fifo_path)
        
        # Frame buffer optimized for real-time performance (single frame buffer)
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        
        # Start libcamera-vid process with optimized parameters for minimal latency
        self.process = subprocess.Popen([
            'libcamera-vid',
            '--width', str(width),
            '--height', str(height),
            '--framerate', str(fps),
            '--timeout', '0',  # Run indefinitely
            '--codec', 'yuv420',
            '--flush',  # Reduce latency
            '--inline',  # Inline headers for reduced latency
            '--listen',  # No preview window
            '--nopreview',  # Explicitly disable any preview
            '--buffer-count', '2',  # Minimal buffer count
            '--output', self.fifo_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Allow camera to initialize
        time.sleep(2)
        
        # Open FIFO for reading
        self.fifo = open(self.fifo_path, 'rb')
        
        # Start frame reading thread
        self.reader_thread = threading.Thread(target=self._frame_reader)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
    def _frame_reader(self):
        """Background thread to continuously read frames from FIFO with optimized processing"""
        while self.running:
            try:
                data = self.fifo.read(self.frame_size)
                if len(data) != self.frame_size:
                    continue
                    
                # Convert YUV420 to BGR with proper reshape for libcamera format
                yuv = np.frombuffer(data, dtype=np.uint8)
                
                # Proper YUV420 format handling for libcamera
                # Y plane: width * height
                # U plane: width * height // 4  
                # V plane: width * height // 4
                y_size = self.width * self.height
                uv_size = y_size // 4
                
                y_plane = yuv[:y_size].reshape(self.height, self.width)
                u_plane = yuv[y_size:y_size + uv_size].reshape(self.height // 2, self.width // 2)
                v_plane = yuv[y_size + uv_size:y_size + 2 * uv_size].reshape(self.height // 2, self.width // 2)
                
                # Reconstruct YUV420 in the format OpenCV expects
                yuv420 = np.zeros((self.height * 3 // 2, self.width), dtype=np.uint8)
                yuv420[:self.height, :] = y_plane
                
                # Interleave U and V planes for I420 format
                uv_interleaved = np.empty((self.height // 2, self.width), dtype=np.uint8)
                uv_interleaved[:, 0::2] = u_plane
                uv_interleaved[:, 1::2] = v_plane
                yuv420[self.height:, :] = uv_interleaved
                
                bgr_frame = cv2.cvtColor(yuv420, cv2.COLOR_YUV2BGR_I420)
                
                # Add to queue (non-blocking, always drop old frames for real-time performance)
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
                if self.running:
                    time.sleep(0.001)  # Minimal delay on error
                break
        
    def read(self):
        """Get the latest frame with minimal timeout for real-time performance"""
        try:
            frame = self.frame_queue.get(timeout=0.1)  # Reduced timeout for better responsiveness
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        """Clean up resources"""
        self.running = False
        try:
            if hasattr(self, 'reader_thread') and self.reader_thread.is_alive():
                self.reader_thread.join(timeout=1.0)
            if hasattr(self, 'fifo'):
                self.fifo.close()
            if hasattr(self, 'process'):
                self.process.terminate()
                self.process.wait(timeout=5)
            if hasattr(self, 'fifo_path') and os.path.exists(self.fifo_path):
                os.unlink(self.fifo_path)
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                os.rmdir(self.temp_dir)
        except Exception as e:
            # Force cleanup if needed
            if hasattr(self, 'process'):
                try:
                    self.process.kill()
                except:
                    pass

def test_camera_devices():
    """Test available camera devices and methods"""
    print("[INFO] Testing camera access methods...")
    
    # Test libcamera subprocess method
    print("[INFO] Testing libcamera subprocess method...")
    try:
        import subprocess
        import tempfile
        import os
        
        # Test if libcamera-vid can capture
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            temp_file = f.name
        
        result = subprocess.run([
            'libcamera-still', '--output', temp_file, '--timeout', '1000',
            '--width', '640', '--height', '480'
        ], capture_output=True, timeout=10)
        
        if result.returncode == 0 and os.path.exists(temp_file):
            # Test if we can read the image
            test_img = cv2.imread(temp_file)
            os.unlink(temp_file)
            if test_img is not None:
                print("[INFO] libcamera subprocess method working")
                return "libcamera", 'libcamera'
        else:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    except Exception as e:
        print(f"[WARN] libcamera subprocess test failed: {e}")
    
    # Test V4L2 direct access
    print("[INFO] Testing V4L2 direct access...")
    video_devices = []
    try:
        import os
        for device in os.listdir('/dev'):
            if device.startswith('video'):
                video_devices.append(f'/dev/{device}')
        
        for device in sorted(video_devices):
            try:
                cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        print(f"[INFO] V4L2 device {device} working")
                        return device, 'v4l2'
                else:
                    cap.release()
            except:
                continue
    except Exception as e:
        print(f"[WARN] V4L2 test failed: {e}")
    
    return None, None

def open_video_capture(width, height, backend):
    """Open video capture with specified backend"""
    
    if backend == 'auto':
        print("[INFO] Auto-detecting best camera backend...")
        device, method = test_camera_devices()
        if method:
            backend = method
            print(f"[INFO] Selected {method} backend")
        else:
            print("[ERROR] No working camera backend found")
            return None

    if backend == 'libcamera':
        print("[INFO] Using libcamera subprocess backend...")
        try:
            cap = LibcameraCapture(width, height)
            return cap
        except Exception as e:
            print(f"[ERROR] libcamera subprocess failed: {e}")
            return None
    
    elif backend == 'v4l2':
        print("[INFO] Using V4L2 backend...")
        video_devices = []
        try:
            import os
            for device in os.listdir('/dev'):
                if device.startswith('video'):
                    video_devices.append(f'/dev/{device}')
            
            for device in sorted(video_devices):
                try:
                    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print(f"[INFO] Using V4L2 device {device}")
                            return cap
                        cap.release()
                except:
                    continue
        except Exception as e:
            print(f"[ERROR] V4L2 backend failed: {e}")
        return None
    
    print(f"[ERROR] Unknown backend: {backend}")
    return None

# ---------------------- TFLite Model Setup ----------------------
def load_tflite_model(model_path):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

# ---------------------- Hand Landmark Inference ----------------------
def run_inference(interpreter, input_details, frame_rgb):
    input_shape = input_details[0]['shape']
    input_tensor = cv2.resize(frame_rgb, (input_shape[2], input_shape[1]))
    input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)
    # Normalize if required (assume 0-255 to 0-1)
    if input_details[0]['dtype'] == np.float32:
        input_tensor = input_tensor / 255.0
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    start = time.time()
    interpreter.invoke()
    inference_time = (time.time() - start) * 1000  # ms
    output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    return output, inference_time

def analyze_model_output(output, debug=False):
    """Analyze and debug model output format"""
    if debug:
        print(f"[DEBUG] Raw output shape: {output.shape}")
        print(f"[DEBUG] Output dtype: {output.dtype}")
        print(f"[DEBUG] Output range: [{output.min():.3f}, {output.max():.3f}]")
        
    # Handle different output formats
    if len(output.shape) == 2:
        landmarks = output.reshape(-1, 3)
    else:
        landmarks = output[0].reshape(-1, 3) if len(output.shape) == 3 else output.reshape(-1, 3)
    
    if debug and landmarks.shape[0] > 0:
        print(f"[DEBUG] Landmarks shape: {landmarks.shape}")
        print(f"[DEBUG] First landmark (wrist): x={landmarks[0][0]:.3f}, y={landmarks[0][1]:.3f}, z={landmarks[0][2]:.3f}")
        
    return landmarks

# ---------------------- Visualization ----------------------
def draw_wrist_only(frame, landmarks, width, height, show_coords=False):
    """Draw only the wrist/palm center landmark for optimized performance"""
    if len(landmarks) == 0:
        return 0
    
    # Only process the first landmark (wrist/palm center)
    x, y, z = landmarks[0][0], landmarks[0][1], landmarks[0][2]
    
    # Ensure coordinates are normalized (0-1 range)
    if x > 1.0 or y > 1.0:
        # If coordinates are not normalized, normalize them by dividing by image dimensions
        x = x / width if x > 1.0 else x
        y = y / height if y > 1.0 else y
    
    cx, cy = int(x * width), int(y * height)
    
    # Clamp coordinates to image bounds
    cx = max(0, min(width - 1, cx))
    cy = max(0, min(height - 1, cy))
    
    # Draw wrist/palm center with larger, more visible circle
    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)  # Larger red circle for wrist
    cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 3)  # White border
    
    # Show coordinates if requested
    if show_coords:
        coord_text = f"Palm: ({cx}, {cy})"
        cv2.putText(frame, coord_text, (cx + 25, cy - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return 1

# ---------------------- Main Loop ----------------------
def run_troubleshooting():
    """Run comprehensive camera troubleshooting"""
    import subprocess
    import os
    
    print("=== CAMERA TROUBLESHOOTING ===")
    
    # Check OS and kernel
    try:
        with open('/etc/os-release', 'r') as f:
            for line in f:
                if line.startswith('PRETTY_NAME='):
                    os_name = line.split('=')[1].strip('"').strip()
                    print(f"OS: {os_name}")
                    break
    except:
        pass
    
    try:
        result = subprocess.run(['uname', '-r'], capture_output=True, text=True)
        print(f"Kernel: {result.stdout.strip()}")
    except:
        pass
    
    # Check camera module status
    print("\n--- Camera Module Status ---")
    try:
        result = subprocess.run(['vcdbg', 'log', 'msg'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            camera_lines = [line for line in result.stdout.split('\n') if 'camera' in line.lower()]
            if camera_lines:
                for line in camera_lines[-5:]:  # Last 5 camera-related messages
                    print(line)
            else:
                print("No camera messages in vcdbg log")
    except:
        print("vcdbg not available (normal on some setups)")
    
    # Check device tree overlays
    print("\n--- Device Tree Configuration ---")
    try:
        with open('/boot/config.txt', 'r') as f:
            content = f.read()
            camera_lines = [line.strip() for line in content.split('\n') 
                           if 'camera' in line.lower() or 'dtoverlay' in line.lower()]
            if camera_lines:
                for line in camera_lines:
                    print(line)
            else:
                print("No camera configuration found in /boot/config.txt")
    except:
        print("Could not read /boot/config.txt")
    
    # Check libcamera
    print("\n--- libcamera Status ---")
    commands = [
        ['libcamera-hello', '--list-cameras'],
        ['libcamera-vid', '--list-cameras', '--timeout', '1'],
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✓ {' '.join(cmd[:2])} working")
                if result.stdout.strip():
                    print(result.stdout.strip())
                break
            else:
                print(f"✗ {' '.join(cmd[:2])} failed: {result.stderr.strip()[:100]}")
        except Exception as e:
            print(f"✗ {' '.join(cmd[:2])} error: {str(e)[:50]}")
    
    # Check video devices
    print("\n--- Video Devices ---")
    video_devices = []
    try:
        for device in os.listdir('/dev'):
            if device.startswith('video'):
                device_path = f'/dev/{device}'
                try:
                    # Try to get device info
                    result = subprocess.run(['v4l2-ctl', '--device', device_path, '--info'], 
                                          capture_output=True, text=True, timeout=3)
                    if result.returncode == 0:
                        if 'Card type' in result.stdout:
                            card_info = result.stdout.split('Card type')[1].split('\n')[0].strip()
                        else:
                            card_info = 'Unknown'
                        print(f"✓ {device_path}: {card_info}")
                    else:
                        print(f"? {device_path}: Could not get info")
                except:
                    print(f"? {device_path}: Present but inaccessible")
                video_devices.append(device_path)
    except:
        pass
    
    if not video_devices:
        print("No /dev/video* devices found")
    
    # Check GStreamer
    print("\n--- GStreamer Status ---")
    try:
        result = subprocess.run(['gst-inspect-1.0', 'libcamerasrc'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ libcamerasrc plugin available")
        else:
            print("✗ libcamerasrc plugin not found")
    except:
        print("✗ GStreamer inspection failed")
    
    # Check permissions
    print("\n--- Permissions ---")
    try:
        import pwd
        username = pwd.getpwuid(os.getuid()).pw_name
        result = subprocess.run(['groups', username], capture_output=True, text=True)
        groups = result.stdout.strip()
        if 'video' in groups:
            print(f"✓ User {username} in video group")
        else:
            print(f"✗ User {username} NOT in video group")
            print("  Fix with: sudo usermod -a -G video $USER")
    except:
        print("Could not check group membership")
    
    print("\n=== END TROUBLESHOOTING ===")
    print("\nRecommended next steps:")
    print("1. If no cameras detected: Check physical connections")
    print("2. If libcamera fails: Try 'sudo raspi-config' > Interface Options > Camera")
    print("3. If permission issues: Run 'sudo usermod -a -G video $USER' then logout/login")
    print("4. If still failing: Try rebooting after enabling camera")
    
def main():
    args = parse_args()
    
    # Run troubleshooting if requested
    if args.troubleshoot:
        run_troubleshooting()
        sys.exit(0)
    
    # Test camera access if requested
    if args.test_camera:
        device, method = test_camera_devices()
        if device is not None:
            print(f"[SUCCESS] Found working camera: {device} using {method}")
        else:
            print("[ERROR] No working camera found")
        sys.exit(0)
    
    width, height = map(int, args.res.split('x'))
    
    # Auto-detect best backend if 'auto' is selected
    if args.backend == 'auto':
        print("[INFO] Auto-detecting best camera backend...")
        device, method = test_camera_devices()
        if method:
            args.backend = method
            print(f"[INFO] Selected {method} backend")
        else:
            print("[ERROR] No working camera backend found")
            sys.exit(1)
    
    cap = open_video_capture(width, height, args.backend)
    
    if cap is None:
        print("[ERROR] Failed to open camera with any available method")
        sys.exit(1)
    
    # Check if model file exists
    import os
    if not os.path.exists(args.model):
        print(f"[ERROR] Model file not found: {args.model}")
        print("[INFO] You can download MediaPipe hand landmark model from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        sys.exit(1)
    
    try:
        interpreter, input_details, output_details = load_tflite_model(args.model)
        if args.debug:
            print(f"[DEBUG] Input shape: {input_details[0]['shape']}")
            print(f"[DEBUG] Output shape: {output_details[0]['shape']}")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)
    
    input_h, input_w = input_details[0]['shape'][1:3]
    prev_time = time.time()
    fps = 0
    frame_count = 0
    display_frame_count = 0  # Counter for display updates
    
    # Frame skipping variables
    skip_counter = 0
    last_wrist_x, last_wrist_y = 0, 0  # Store last known wrist position
    last_inf_time = 0  # Store last inference time
    
    if args.headless:
        print("[INFO] Running in headless mode. Press Ctrl+C to exit.")
    else:
        print("[INFO] Press ESC to exit.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break
            
            frame_count += 1
            
            # Skip frames for performance optimization
            if skip_counter < args.frame_skip:
                skip_counter += 1
                
                # Display the previous wrist position for skipped frames
                if last_wrist_x > 0 and last_wrist_y > 0:
                    cv2.circle(frame, (last_wrist_x, last_wrist_y), 15, (0, 0, 255), -1)
                    cv2.circle(frame, (last_wrist_x, last_wrist_y), 18, (255, 255, 255), 3)
                    
                    if not args.headless:
                        coord_text = f"Palm: ({last_wrist_x}, {last_wrist_y})"
                        cv2.putText(frame, coord_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (0, 255, 0), 2)
                
                # FPS calculation for all frames
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time)
                prev_time = curr_time
                
                if not args.headless:
                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    cv2.putText(frame, f"Inference: {last_inf_time:.1f} ms (cached)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    cv2.imshow('Hand Tracking - Optimized (Wrist Only)', frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                continue
            
            # Reset skip counter and process this frame
            skip_counter = 0
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output, inf_time = run_inference(interpreter, input_details, frame_rgb)
            last_inf_time = inf_time
            
            # Handle different output formats - only extract wrist (first landmark)
            if len(output.shape) == 2:
                landmarks = output.reshape(-1, 3)
            else:
                landmarks = output[0].reshape(-1, 3) if len(output.shape) == 3 else output.reshape(-1, 3)
            
            if landmarks.shape[0] >= 1:  # Only need first landmark (wrist)
                landmarks_drawn = draw_wrist_only(frame, landmarks[:1], width, height, args.show_coords)
                
                # Get normalized wrist coordinates
                wrist_x_norm, wrist_y_norm = landmarks[0][0], landmarks[0][1]
                
                # Normalize if needed
                if wrist_x_norm > 1.0 or wrist_y_norm > 1.0:
                    wrist_x_norm = wrist_x_norm / width if wrist_x_norm > 1.0 else wrist_x_norm
                    wrist_y_norm = wrist_y_norm / height if wrist_y_norm > 1.0 else wrist_y_norm
                
                wrist_x, wrist_y = int(wrist_x_norm * width), int(wrist_y_norm * height)
                last_wrist_x, last_wrist_y = wrist_x, wrist_y  # Store for skipped frames
                
                # Always display wrist coordinates on frame (even in non-debug mode)
                if not args.headless:
                    coord_text = f"Palm: ({wrist_x}, {wrist_y})"
                    cv2.putText(frame, coord_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (0, 255, 0), 2)
                
                if args.debug:
                    print(f"Palm: ({wrist_x}, {wrist_y}) [norm: {wrist_x_norm:.3f}, {wrist_y_norm:.3f}], "
                          f"Inference: {inf_time:.1f} ms, Frame: {frame_count}")
            elif args.debug:
                print(f"[DEBUG] No landmarks detected, shape: {landmarks.shape}")
            
            # FPS calculation
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time
            
            if not args.headless:
                # Update display every frame for smooth visualization
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Inference: {inf_time:.1f} ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"Skip: {args.frame_skip}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Hand Tracking - Optimized (Wrist Only)', frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                display_frame_count += 1
            else:
                # In headless mode, print periodic updates
                if frame_count % 30 == 0:  # Every 30 frames
                    print(f"Frame {frame_count}: FPS={fps:.1f}, Inference={inf_time:.1f}ms")
                
                # Small delay to prevent excessive CPU usage in headless mode
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] Runtime error: {e}")
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
