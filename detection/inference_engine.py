"""
TensorFlow Lite model inference engine.
"""

import cv2
import numpy as np

class InferenceEngine:
    """Handles inference using the TensorFlow Lite model."""
    
    def __init__(self, model_loader):
        """Initialize inference engine with a model loader instance."""
        self.model_loader = model_loader
        self.original_frame_shape = None  # Will store (height, width) of original frame
        
        self.min_box_area = 0.02  # Minimum normalized box area to accept detection
        
    def set_frame_dimensions(self, frame_shape):
        """Set original frame dimensions for coordinate conversion."""
        self.original_frame_shape = frame_shape[:2]  # (height, width)
        
    def prepare_input(self, frame):
        """Prepare frame for model input."""
        # Store original frame dimensions for coordinate conversion
        self.original_frame_shape = frame.shape[:2]  # (height, width)
        
        # Get target size from model input details
        target_height, target_width = self.model_loader.get_input_shape()
        
        # Resize frame to match model input size
        input_frame = cv2.resize(frame, (target_width, target_height))
        
        # Normalize to float32 [0,1]
        input_frame = np.float32(input_frame) / 255.0
        
        # Add batch dimension
        input_tensor = np.expand_dims(input_frame, axis=0)
        
        return input_tensor
        
    def run_inference(self, input_tensor):
        """Run palm detection only and return a single wrist landmark."""
        interpreter = self.model_loader.interpreter
        input_details = self.model_loader.input_details
        output_details = self.model_loader.output_details
        
        # Run palm detection
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        
        # Retrieve outputs
        kp_out = interpreter.get_tensor(output_details[0]['index'])  # [1, N, 18]
        sc_out = interpreter.get_tensor(output_details[1]['index'])  # [1, N, 1]
        
        # Find best detection
        scores = sc_out[0].flatten()
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        
        # Compute bounding box from palm keypoints
        kpts = kp_out[0, best_idx].reshape(-1, 2)
        xmin, xmax = kpts[:,0].min(), kpts[:,0].max()
        ymin, ymax = kpts[:,1].min(), kpts[:,1].max()
        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        
        # Normalize coordinates to [0,1] relative to model input size
        model_h, model_w = self.model_loader.get_input_shape()
        norm_x = cx / model_w
        norm_y = cy / model_h
        # Build output: single normalized landmark and confidence
        landmarks = np.array([[[norm_x, norm_y, 0.0]]])
        hand_scores = np.array([[best_score]])
        return landmarks, None, hand_scores, None
    
    def _create_landmarks_from_palm(self, palm_keypoints, bbox=None):
        """Create 21 hand landmarks from palm keypoints with enhanced precision."""
        # Palm detection typically gives us 7-21 keypoints
        # We'll map these to the standard 21 MediaPipe hand landmarks
        
        # DEBUG: Print raw palm keypoints for analysis (can be removed in production)
        # print(f"[COORD_DEBUG] Raw palm_keypoints length: {len(palm_keypoints)}")
        # print(f"[COORD_DEBUG] Raw palm_keypoints: {palm_keypoints}")
        
        full_landmarks = np.zeros((21, 3))
        
        # MediaPipe palm detection typically gives us 7 or 9 keypoints
        # Need to determine the format and properly extract coordinates
        
        # Try different common formats for palm keypoint extraction
        if len(palm_keypoints) == 18:  # 9 keypoints with x,y coordinates flattened
            # This is the most common format - 9 palm keypoints
            # Reshape to get proper coordinate pairs
            reshaped_keypoints = palm_keypoints.reshape(-1, 2)
            
            # DEBUG: Show reshaped keypoints for analysis (can be removed in production)
            # print(f"[COORD_DEBUG] Reshaped keypoints shape: {reshaped_keypoints.shape}")
            # print(f"[COORD_DEBUG] Reshaped keypoints: {reshaped_keypoints}")
            
            # MediaPipe palm detection keypoint order (typically):
            # 0: Wrist center
            # 1: Thumb base
            # 2: Index finger base
            # 3: Middle finger base
            # 4: Ring finger base
            # 5: Pinky base
            # 6,7,8: Additional palm points
            
            # Map palm keypoints to hand landmarks with known correspondence
            key_mappings = [
                (0, 0),  # Wrist center → Wrist
                (1, 1),  # Thumb base → Thumb base
                (2, 5),  # Index base → Index base
                (3, 9),  # Middle base → Middle base
                (4, 13), # Ring base → Ring base
                (5, 17), # Pinky base → Pinky base
            ]
            
            # Apply mappings for known correspondences
            for src_idx, dst_idx in key_mappings:
                if src_idx < len(reshaped_keypoints):
                    x, y = reshaped_keypoints[src_idx]
                    
                    # DEBUG: Show coordinate transformation (can be removed in production)
                    # print(f"[COORD_DEBUG] Mapping keypoint {src_idx} -> landmark {dst_idx}: ({x:.3f}, {y:.3f})")
                    
                    # Convert from model coordinate space to pixel coordinates
                    try:
                        pixel_x, pixel_y = self._convert_model_coords_to_pixels(x, y)
                    except Exception as e:
                        print(f"[WARNING] Error in coordinate conversion: {e}")
                        pixel_x, pixel_y = int(x), int(y)  # Fallback
                    
                    full_landmarks[dst_idx] = [pixel_x, pixel_y, 0.0]
                    
            # Create finger landmarks by extending from bases toward estimated fingertips
            self._extend_finger_landmarks(full_landmarks, bbox)
            
        elif len(palm_keypoints) == 14:  # 7 keypoints format
            # Alternative format with 7 keypoints
            reshaped_keypoints = palm_keypoints.reshape(-1, 2)
            
            # Map available keypoints
            key_mappings = [
                (0, 0),  # Wrist
                (1, 1),  # Thumb base
                (2, 5),  # Index base
                (3, 9),  # Middle base
                (4, 13), # Ring base
                (5, 17), # Pinky base
            ]
            
            for src_idx, dst_idx in key_mappings:
                if src_idx < len(reshaped_keypoints):
                    x, y = reshaped_keypoints[src_idx]
                    full_landmarks[dst_idx] = [x, y, 0.0]
                    
            # Create finger landmarks by extending from bases
            self._extend_finger_landmarks(full_landmarks, bbox)
            
        else:
            # For other formats, fall back to bounding box approach if available
            if bbox is not None:
                return self._create_landmarks_from_box(bbox)
            
            # If no bbox, try to extract as many valid keypoints as possible
            num_keypoints = len(palm_keypoints) // 2
            for i in range(min(num_keypoints, 21)):
                if i * 2 + 1 < len(palm_keypoints):
                    x = palm_keypoints[i * 2]
                    y = palm_keypoints[i * 2 + 1]
                    full_landmarks[i] = [x, y, 0.0]
            
            # Fill remaining landmarks with interpolated values
            for i in range(num_keypoints, 21):
                if num_keypoints > 0:
                    # Use the last available keypoint as reference
                    ref_idx = min(i, num_keypoints - 1)
                    full_landmarks[i] = full_landmarks[ref_idx].copy()
                    # Add small offset to avoid identical points
                    full_landmarks[i][0] += (i - ref_idx) * 0.01
                    full_landmarks[i][1] += (i - ref_idx) * 0.01
        
        return full_landmarks
        
    def _convert_model_coords_to_pixels(self, x, y):
        """Convert coordinates from model space to pixel space."""
        if self.original_frame_shape is None:
            return int(x), int(y)

        original_height, original_width = self.original_frame_shape
        model_height, model_width = self.model_loader.get_input_shape()

        # Scale x directly from model to frame space
        pixel_x = int((x / model_width) * original_width)
        # Scale y directly (model y and frame y both increase downward)
        pixel_y = int((y / model_height) * original_height)
        # debug conversion log removed

        return pixel_x, pixel_y
        
    def _extend_finger_landmarks(self, landmarks, bbox=None):
        """Create finger landmarks by extending from base positions."""
        # For each finger, we know the base position and need to estimate joints and tip
        # We'll use the relative positions of fingers in a typical hand pose
        
        # Get palm center (average of bases)
        bases = [1, 5, 9, 13, 17]  # Thumb and finger bases
        valid_bases = []
        for idx in bases:
            if not np.all(landmarks[idx] == 0):
                valid_bases.append(landmarks[idx])
                
        if not valid_bases:
            return
            
        palm_center = np.mean(valid_bases, axis=0)
        
        # Determine overall hand orientation and scale from the base points
        hand_width = 0
        hand_height = 0
        
        if len(valid_bases) > 1:
            min_x = min(p[0] for p in valid_bases)
            max_x = max(p[0] for p in valid_bases) 
            min_y = min(p[1] for p in valid_bases)
            max_y = max(p[1] for p in valid_bases)
            hand_width = max(0.001, max_x - min_x)
            hand_height = max(0.001, max_y - min_y)
        elif bbox is not None:
            # Use bbox for scale if available
            ymin, xmin, ymax, xmax = bbox
            hand_width = max(0.001, xmax - xmin)
            hand_height = max(0.001, ymax - ymin)
        else:
            # Default scale if no information available
            hand_width = 0.2
            hand_height = 0.3
            
        # Relative proportions for finger joints based on typical hand anatomy
        finger_relative_lengths = [
            [0.3, 0.6, 1.0],    # Thumb: base→mid→tip (shorter)
            [0.33, 0.66, 1.0],  # Index: base→mid→tip
            [0.33, 0.66, 1.0],  # Middle: base→mid→tip (longest)
            [0.33, 0.66, 1.0],  # Ring: base→mid→tip
            [0.33, 0.66, 1.0],  # Pinky: base→mid→tip (shortest)
        ]
        
        # For each finger, extend from base to create other joints
        for finger_idx, base_idx in enumerate([1, 5, 9, 13, 17]):
            if np.all(landmarks[base_idx] == 0):
                continue  # Skip if base position is not available
                
            # Get the base position
            base_pos = landmarks[base_idx]
            
            # Calculate finger direction vector - from palm center to base
            direction = base_pos - palm_center
            
            # Normalize and scale for finger length
            finger_length = max(hand_height * 0.4, hand_width * 0.3)  # Estimated finger length
            
            # Avoid division by zero
            if np.linalg.norm(direction) > 0.001:
                direction = direction / np.linalg.norm(direction)
            else:
                # Default direction if calculation fails
                direction = np.array([0, -1, 0])  # Point upward
            
            # Create the remaining joints by extending along the direction vector
            for joint_idx, rel_length in enumerate(finger_relative_lengths[finger_idx]):
                # Calculate target index in the 21-point hand model
                target_idx = base_idx + joint_idx + 1
                
                # Only update if the point hasn't been set already
                if target_idx < 21 and np.all(landmarks[target_idx] == 0):
                    # Calculate joint position by extending from base
                    joint_pos = base_pos + direction * (rel_length * finger_length)
                    landmarks[target_idx] = joint_pos
    
    def _create_landmarks_from_box(self, bbox):
        """Create landmarks from bounding box (fallback method)."""
        # bbox format: [ymin, xmin, ymax, xmax] normalized
        ymin, xmin, ymax, xmax = bbox
        
        # Calculate palm center and size
        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        
        # Create 21 landmarks in a hand-like pattern
        landmarks = np.zeros((21, 3))
        
        # Wrist (landmark 0)
        landmarks[0] = [center_x, ymax - height * 0.1, 0.0]
        
        # Thumb (landmarks 1-4)
        for i in range(1, 5):
            landmarks[i] = [
                center_x - width * 0.3 + (i-1) * width * 0.1,
                center_y - height * 0.2 + (i-1) * height * 0.15,
                0.0
            ]
        
        # Index finger (landmarks 5-8)
        for i in range(5, 9):
            landmarks[i] = [
                center_x - width * 0.2 + (i-5) * width * 0.05,
                center_y - height * 0.4 + (i-5) * height * 0.2,
                0.0
            ]
        
        # Middle finger (landmarks 9-12)
        for i in range(9, 13):
            landmarks[i] = [
                center_x + (i-9) * width * 0.05,
                center_y - height * 0.45 + (i-9) * height * 0.25,
                0.0
            ]
        
        # Ring finger (landmarks 13-16)
        for i in range(13, 17):
            landmarks[i] = [
                center_x + width * 0.2 + (i-13) * width * 0.05,
                center_y - height * 0.4 + (i-13) * height * 0.2,
                0.0
            ]
        
        # Pinky (landmarks 17-20)
        for i in range(17, 21):
            landmarks[i] = [
                center_x + width * 0.3 + (i-17) * width * 0.05,
                center_y - height * 0.3 + (i-17) * height * 0.15,
                0.0
            ]
        
        return landmarks
    
    def _create_dummy_landmarks(self):
        """Create dummy landmarks when no detection data is available."""
        # Create a simple hand pattern in the center of the image
        landmarks = np.zeros((21, 3))
        center_x, center_y = 0.5, 0.5  # Normalized center
        
        # Create a basic hand outline
        for i in range(21):
            angle = i * 2 * np.pi / 21
            radius = 0.1
            landmarks[i] = [
                center_x + radius * np.cos(angle),
                center_y + radius * np.sin(angle),
                0.0
            ]
        
        return landmarks
