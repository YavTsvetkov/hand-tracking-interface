"""
Camera interface using libcamera for Raspberry Pi.
"""

import os
import tempfile
import subprocess
import queue
import threading
import cv2
import numpy as np
import time

class LibcameraCapture:
    """Libcamera capture implementation for Raspberry Pi camera."""
    
    def __init__(self, width, height, fps=30):
        """Initialize libcamera capture with specified resolution and framerate."""
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_size = width * height * 3 // 2  # YUV420
        
        # Create FIFO for camera data
        self.temp_dir = tempfile.mkdtemp()
        self.fifo_path = os.path.join(self.temp_dir, 'camera_fifo')
        os.mkfifo(self.fifo_path)
        
        self.frame_queue = queue.Queue(maxsize=1)
        self.running = True
        self._setup_process()
        
    def _setup_process(self):
        """Start libcamera-vid process with optimized settings."""
        self.process = subprocess.Popen([
            'libcamera-vid', 
            '--width', str(self.width), 
            '--height', str(self.height),
            '--framerate', str(self.fps), 
            '--timeout', '0', 
            '--codec', 'yuv420',
            '--flush',  # Reduce latency
            '--inline',  # Inline headers for reduced latency
            '--output', self.fifo_path,
            '--nopreview'  # For headless operation
        ])
        
        # Start reader thread
        self.reader_thread = threading.Thread(target=self._read_frames)
        self.reader_thread.daemon = True
        self.reader_thread.start()
    
    def _read_frames(self):
        """Background thread to read frames from FIFO."""
        try:
            with open(self.fifo_path, 'rb') as fifo:
                while self.running:
                    # Read YUV420 data from FIFO
                    yuv_data = fifo.read(self.frame_size)
                    if not yuv_data or len(yuv_data) != self.frame_size:
                        if self.running:  # Only log if expected to be running
                            print("[ERROR] Incomplete frame data")
                        continue

                    # Convert YUV420 to BGR for OpenCV processing
                    yuv = np.frombuffer(yuv_data, dtype=np.uint8)
                    yuv = yuv.reshape((self.height * 3 // 2, self.width))
                    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
                    
                    # Put in queue, replacing old frame if exists
                    try:
                        self.frame_queue.put(bgr, block=False)
                    except queue.Full:
                        # Clear old frame and put new one
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(bgr, block=False)
                        except (queue.Empty, queue.Full):
                            pass  # Rare race condition, just continue
        except Exception as e:
            if self.running:  # Only log if expected to be running
                print(f"[ERROR] Camera reader thread: {e}")
    
    def read(self):
        """Read a frame from the camera, similar to cv2.VideoCapture.read()."""
        try:
            frame = self.frame_queue.get(timeout=2.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        """Release camera resources."""
        self.running = False
        
        # Terminate libcamera process
        if hasattr(self, 'process'):
            self.process.terminate()
            self.process.wait(timeout=2.0)
            
        # Cleanup temporary directory and FIFO
        if os.path.exists(self.fifo_path):
            try:
                os.unlink(self.fifo_path)
            except OSError:
                pass
                
        if os.path.exists(self.temp_dir):
            try:
                os.rmdir(self.temp_dir)
            except OSError:
                pass
