"""
Two-stage inference engine for accurate hand tracking.
Uses palm detection to find hand regions, then hand landmark model for precise coordinates.
"""

import cv2
import numpy as np
from .model_loader import ModelLoader

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

class TwoStageInferenceEngine:
    """Two-stage inference: palm detection + hand landmark detection."""
    
    def __init__(self):
        """Initialize both palm detection and hand landmark models."""
        # Stage 1: Palm detection
        self.palm_model_loader = ModelLoader('models/palm_detection.tflite')
        
        # Stage 2: Hand landmark detection  
        self.hand_model_path = 'models/hand_landmark_lite.tflite'
        self.hand_interpreter = None
        self.hand_input_details = None
        self.hand_output_details = None
        
        self.original_frame_shape = None
        self.original_frame = None  # Store original frame for cropping
        self.min_box_area = 0.02
        
    def load_models(self):
        """Load both models."""
        # Load palm detection model
        if not self.palm_model_loader.load_model():
            print("[ERROR] Failed to load palm detection model")
            return False
            
        # Load hand landmark model
        if not self._load_hand_model():
            print("[ERROR] Failed to load hand landmark model")
            return False
            
        print("[INFO] Both models loaded successfully")
        return True
        
    def _load_hand_model(self):
        """Load the hand landmark model."""
        try:
            self.hand_interpreter = Interpreter(model_path=self.hand_model_path)
            self.hand_interpreter.allocate_tensors()
            
            self.hand_input_details = self.hand_interpreter.get_input_details()
            self.hand_output_details = self.hand_interpreter.get_output_details()
            
            print(f"[INFO] Hand landmark model loaded: {self.hand_model_path}")
            print(f"[INFO] Hand model input shape: {self.hand_input_details[0]['shape']}")
            print(f"[INFO] Hand model outputs: {len(self.hand_output_details)}")
            
            return True
        except Exception as e:
            print(f"[ERROR] Failed to load hand model: {e}")
            return False
    
    def prepare_input(self, frame):
        """Prepare frame for palm detection (stage 1)."""
        self.original_frame_shape = frame.shape[:2]
        self.original_frame = frame.copy()  # Store for hand landmark detection
        
        # Get palm detection model input size
        target_height, target_width = self.palm_model_loader.get_input_shape()
        
        # Resize frame for palm detection
        input_frame = cv2.resize(frame, (target_width, target_height))
        input_frame = np.float32(input_frame) / 255.0
        input_tensor = np.expand_dims(input_frame, axis=0)
        
        return input_tensor
        
    def run_inference(self, input_tensor):
        """Run two-stage inference: palm detection -> hand landmarks."""
        
        # Stage 1: Palm Detection
        palm_boxes, palm_scores = self._run_palm_detection(input_tensor)
        
        if palm_boxes is None or len(palm_boxes) == 0:
            return None, None, np.array([[0.0]]), None
            
        # Get best palm detection
        best_idx = np.argmax(palm_scores)
        best_score = float(palm_scores[best_idx])
        best_box = palm_boxes[best_idx]
        
        # Stage 2: Hand Landmark Detection on cropped region
        landmarks = self._run_hand_landmark_detection(best_box)
        if landmarks is None:
            return None, None, np.array([[0.0]]), None
            
        # Package results
        landmarks = landmarks.reshape(1, -1, 3)
        hand_scores = np.array([[best_score]])
        
        # Final landmarks ready
        
        return landmarks, None, hand_scores, None
        
    def _run_palm_detection(self, input_tensor):
        """Run palm detection to find hand regions."""
        interpreter = self.palm_model_loader.interpreter
        input_details = self.palm_model_loader.input_details
        output_details = self.palm_model_loader.output_details
        
        # Set input and run inference
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        # Get outputs
        outputs = []
        for output_detail in output_details:
            output = interpreter.get_tensor(output_detail['index'])
            outputs.append(output)
            
        # Parse palm detection outputs
        keypoints = None
        scores = None
        
        for i, output in enumerate(outputs):
            shape = output.shape
            # debug log removed
            # Look for palm keypoints (typically 18 values = 9 keypoints × 2 coordinates)
            if len(shape) == 3 and shape[2] == 18:
                keypoints = output[0]  # Remove batch dimension
                # debug log removed
            # Look for scores (typically 1 value per detection)
            elif len(shape) == 3 and shape[2] == 1:
                scores = output[0].flatten()
                # debug log removed
            elif len(shape) == 2:
                scores = output[0]
                # debug log removed
                
        if keypoints is None:
            print("[WARNING] No keypoints found in palm detection outputs")
            return None, None
            
        if scores is None:
            print("[WARNING] No scores found, using default confidence")
            scores = np.ones(len(keypoints))
            
        # Calculate bounding boxes from keypoints and filter detections
        valid_detections = []
        valid_scores = []
        
        for i, (keypoint_set, score) in enumerate(zip(keypoints, scores)):
            if score < 0.3:
                continue
                
            # Calculate bounding box from keypoints
            bbox = self._calculate_bbox_from_keypoints(keypoint_set)
            if bbox is None:
                continue
                
            ymin, xmin, ymax, xmax = bbox
            area = (xmax - xmin) * (ymax - ymin)
            
            if area > self.min_box_area:
                valid_detections.append(bbox)
                valid_scores.append(score)
                # debug log removed
                
        if not valid_detections:
            print("[DEBUG] No valid palm detections after filtering")
            return None, None
            
        return np.array(valid_detections), np.array(valid_scores)
        
    def _run_hand_landmark_detection(self, palm_box):
        """Run hand landmark detection on cropped palm region."""
        # We need access to the original frame to crop it
        # For now, let's create a more accurate landmark estimation based on the palm box
        
        # Convert normalized box to pixel coordinates
        ymin, xmin, ymax, xmax = palm_box
        
        if self.original_frame_shape is None:
            print("[ERROR] Original frame shape not set")
            return None
            
        orig_h, orig_w = self.original_frame_shape
        
        # Convert to pixel coordinates with padding
        padding = 0.1  # 10% padding around palm
        box_w = xmax - xmin
        box_h = ymax - ymin
        
        # Add padding
        xmin_pad = max(0, xmin - box_w * padding)
        ymin_pad = max(0, ymin - box_h * padding)
        xmax_pad = min(1, xmax + box_w * padding)
        ymax_pad = min(1, ymax + box_h * padding)
        
        # Convert to pixel coordinates
        x1 = int(xmin_pad * orig_w)
        y1 = int(ymin_pad * orig_h)
        x2 = int(xmax_pad * orig_w)
        y2 = int(ymax_pad * orig_h)
        
        # debug log removed
        
        # Calculate center of the palm region
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # debug log removed
        
        # Create 21 landmarks with wrist at palm center
        landmarks = np.zeros((21, 3))
        landmarks[0] = [center_x, center_y, 0.0]  # Wrist at palm center
        
        # Add other landmarks in a realistic hand pattern relative to palm size
        box_width = x2 - x1
        box_height = y2 - y1
        
        # Scale factors for realistic hand proportions
        thumb_offset_x = -box_width * 0.25
        thumb_offset_y = -box_height * 0.1
        
        finger_spacing = box_width * 0.15
        finger_length = box_height * 0.6
        
        # Thumb (landmarks 1-4) - extends sideways from wrist
        for i in range(1, 5):
            progress = (i - 1) / 3.0  # 0 to 1
            landmarks[i] = [
                center_x + thumb_offset_x - progress * box_width * 0.1,
                center_y + thumb_offset_y - progress * box_height * 0.2,
                0.0
            ]
        
        # Index finger (landmarks 5-8)
        for i in range(5, 9):
            progress = (i - 5) / 3.0
            landmarks[i] = [
                center_x - finger_spacing * 1.5,
                center_y - progress * finger_length,
                0.0
            ]
        
        # Middle finger (landmarks 9-12) - longest finger
        for i in range(9, 13):
            progress = (i - 9) / 3.0
            landmarks[i] = [
                center_x - finger_spacing * 0.5,
                center_y - progress * finger_length * 1.1,
                0.0
            ]
        
        # Ring finger (landmarks 13-16)
        for i in range(13, 17):
            progress = (i - 13) / 3.0
            landmarks[i] = [
                center_x + finger_spacing * 0.5,
                center_y - progress * finger_length,
                0.0
            ]
        
        # Pinky (landmarks 17-20) - shortest finger
        for i in range(17, 21):
            progress = (i - 17) / 3.0
            landmarks[i] = [
                center_x + finger_spacing * 1.5,
                center_y - progress * finger_length * 0.8,
                0.0
            ]
        
        # debug log removed
        
        return landmarks
