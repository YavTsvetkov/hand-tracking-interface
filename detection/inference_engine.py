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
        self.processed_frame_shape = None # Will store (height, width) of the frame *after* FrameProcessor.preprocess
        self.crop_offset_x = 0
        self.crop_offset_y = 0
        
        self.min_box_area = 0.02  # Minimum normalized box area to accept detection
        
    def set_frame_dimensions(self, frame_shape, processed_frame_shape, crop_offset_x=0, crop_offset_y=0):
        """Set original and processed frame dimensions and crop offsets."""
        self.original_frame_shape = frame_shape[:2]  # (height, width)
        self.processed_frame_shape = processed_frame_shape[:2] # (height, width)
        self.crop_offset_x = crop_offset_x
        self.crop_offset_y = crop_offset_y
        
    def prepare_input(self, processed_frame):
        """Prepare frame (already processed by FrameProcessor) for model input."""
        # Store processed frame dimensions for coordinate conversion
        # self.original_frame_shape is set by set_frame_dimensions
        self.processed_frame_shape = processed_frame.shape[:2] # (height, width)
        
        # Get target size from model input details
        target_height, target_width = self.model_loader.get_input_shape()

        # --- Aspect Ratio Preserving Resize (Letterboxing/Pillarboxing) ---
        proc_h, proc_w = self.processed_frame_shape
        
        # Calculate scaling factor to fit within target_width, target_height
        scale_w = target_width / proc_w
        scale_h = target_height / proc_h
        scale = min(scale_w, scale_h) # Use the smaller scale to fit entirely

        # New dimensions after scaling
        new_w, new_h = int(proc_w * scale), int(proc_h * scale)

        # Resize with aspect ratio preservation
        resized_frame = cv2.resize(processed_frame, (new_w, new_h))

        # Create a black canvas of target size
        input_frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        # Calculate padding
        pad_x = (target_width - new_w) // 2
        pad_y = (target_height - new_h) // 2
        
        # Place resized frame onto the canvas
        input_frame[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_frame
        # --- End Aspect Ratio Preserving Resize ---
        
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
        
        # Enhanced debugging for coordinate issues
        print(f"[DEBUG] === INFERENCE DEBUGGING ===")
        print(f"Original Frame shape (from main): {self.original_frame_shape}")
        print(f"Processed Frame shape (after crop/color, before resize): {self.processed_frame_shape}")
        print(f"Crop offsets (x,y): ({self.crop_offset_x}, {self.crop_offset_y})")
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Find best detection
        scores = sc_out[0].flatten()
        best_idx = np.argmax(scores)
        best_score = float(scores[best_idx])
        
        # Check if we have a valid detection above minimum confidence
        min_confidence = 0.3  # Minimum confidence threshold
        if best_score < min_confidence:
            print(f"[DEBUG] Low confidence detection {best_score:.4f} < {min_confidence}, returning no detection")
            return None, None, None, None
        
        # Raw model outputs analysis
        print(f"Raw kp_out shape: {kp_out.shape}, range: [{kp_out.min():.4f}, {kp_out.max():.4f}]")
        print(f"Raw sc_out shape: {sc_out.shape}, range: [{sc_out.min():.4f}, {sc_out.max():.4f}]")
        print(f"Best detection: idx={best_idx}, score={best_score:.4f}")
        
        # Extract and analyze raw keypoints
        raw_keypoints = kp_out[0, best_idx]
        print(f"Raw best keypoints: {raw_keypoints}")
        
        # Reshape keypoints
        kpts = raw_keypoints.reshape(-1, 2)
        print(f"Reshaped keypoints shape: {kpts.shape}")
        print(f"First 3 keypoints: {kpts[:3]}")
        print(f"All keypoints X values: {kpts[:, 0]}")
        print(f"All keypoints Y values: {kpts[:, 1]}")
        
        # Check if all keypoints are identical (invalid detection)
        x_variance = np.var(kpts[:, 0])
        y_variance = np.var(kpts[:, 1])
        print(f"Keypoint variance - X: {x_variance:.6f}, Y: {y_variance:.6f}")
        
        if x_variance < 1e-6 and y_variance < 1e-6:
            print(f"[DEBUG] All keypoints identical (variance too low), likely invalid detection")
            return None, None, None, None
        
        # Before clipping analysis
        print(f"Before clipping - X range: [{kpts[:,0].min():.4f}, {kpts[:,0].max():.4f}], Y range: [{kpts[:,1].min():.4f}, {kpts[:,1].max():.4f}]")
        
        # Don't clip aggressively - let's see the raw values first
        # Only clip extreme outliers beyond reasonable bounds
        kpts[:, 0] = np.clip(kpts[:, 0], -0.5, 1.5)  # Allow some out-of-bounds values
        kpts[:, 1] = np.clip(kpts[:, 1], -0.5, 1.5)
        print(f"After gentle clipping - X range: [{kpts[:,0].min():.4f}, {kpts[:,0].max():.4f}], Y range: [{kpts[:,1].min():.4f}, {kpts[:,1].max():.4f}]")
        
        # Try different palm center calculation methods
        xmin, xmax = kpts[:,0].min(), kpts[:,0].max()
        ymin, ymax = kpts[:,1].min(), kpts[:,1].max()
        cx_minmax = (xmin + xmax) / 2.0
        cy_minmax = (ymin + ymax) / 2.0
        
        # Alternative calculations
        cx_avg = np.mean(kpts[:,0])
        cy_avg = np.mean(kpts[:,1])
        cx_median = np.median(kpts[:,0])
        cy_median = np.median(kpts[:,1])
        cx_first = kpts[0, 0]
        cy_first = kpts[0, 1]
        
        print(f"Coordinate calculation methods:")
        print(f"  Min/Max: ({cx_minmax:.4f}, {cy_minmax:.4f})")
        print(f"  Average: ({cx_avg:.4f}, {cy_avg:.4f})")
        print(f"  Median: ({cx_median:.4f}, {cy_median:.4f})")
        print(f"  First keypoint: ({cx_first:.4f}, {cy_first:.4f})")
        
        # Use the method that gives most variation (not stuck at 0.5, 0.5)
        # If variance is very low, try using first keypoint instead
        if x_variance < 0.01 and y_variance < 0.01:
            print(f"[DEBUG] Low variance detected, using first keypoint")
            cx, cy = cx_first, cy_first
        else:
            print(f"[DEBUG] Using average method")
            cx, cy = cx_avg, cy_avg
        
        # Normalize to [0,1] range for output
        cx = np.clip(cx, 0.0, 1.0)
        cy = np.clip(cy, 0.0, 1.0)
        
        # Pixel conversion preview
        if self.processed_frame_shape: # Use processed_frame_shape for preview as it's the basis for model input
            # Get model input dimensions (where cx, cy are normalized)
            model_h, model_w = self.model_loader.get_input_shape()
            
            # Calculate scale and padding used in prepare_input
            proc_h, proc_w = self.processed_frame_shape
            scale_w_fit = model_w / proc_w
            scale_h_fit = model_h / proc_h
            scale_fit = min(scale_w_fit, scale_h_fit)
            
            new_proc_w_on_model = int(proc_w * scale_fit)
            new_proc_h_on_model = int(proc_h * scale_fit)
            
            pad_x_on_model = (model_w - new_proc_w_on_model) // 2
            pad_y_on_model = (model_h - new_proc_h_on_model) // 2

            # Convert model's normalized cx, cy (0-1 relative to model input)
            # back to coordinates relative to the *resized_frame* (the non-padded part)
            # Model output (cx, cy) is normalized to model_w, model_h
            # Step 1: Denormalize from model input (e.g., 192x192)
            abs_x_on_model = cx * model_w
            abs_y_on_model = cy * model_h
            
            # Step 2: Account for padding
            x_on_resized_proc_frame = abs_x_on_model - pad_x_on_model
            y_on_resized_proc_frame = abs_y_on_model - pad_y_on_model
            
            # Step 3: Normalize relative to the resized_proc_frame dimensions
            # (new_proc_w_on_model, new_proc_h_on_model)
            # Avoid division by zero if new_proc_w/h_on_model is zero (should not happen with valid image)
            norm_x_on_proc_frame = x_on_resized_proc_frame / new_proc_w_on_model if new_proc_w_on_model > 0 else 0
            norm_y_on_proc_frame = y_on_resized_proc_frame / new_proc_h_on_model if new_proc_h_on_model > 0 else 0
            
            # Clip to [0,1] as these are normalized to processed_frame
            norm_x_on_proc_frame = np.clip(norm_x_on_proc_frame, 0.0, 1.0)
            norm_y_on_proc_frame = np.clip(norm_y_on_proc_frame, 0.0, 1.0)

            # Now, norm_x_on_proc_frame and norm_y_on_proc_frame are normalized coordinates
            # with respect to the self.processed_frame_shape (the frame after cropping but before letterboxing)
            # These are the coordinates that main.py expects.
            
            # For the debug print here, scale to original_frame_shape for a full preview
            # This involves scaling from processed_frame to original_frame and adding crop offsets
            final_preview_x = int(norm_x_on_proc_frame * self.processed_frame_shape[1] + self.crop_offset_x)
            final_preview_y = int(norm_y_on_proc_frame * self.processed_frame_shape[0] + self.crop_offset_y)

            print(f"  Model cx,cy: ({cx:.4f}, {cy:.4f})")
            print(f"  Normalized on processed frame (for main.py): ({norm_x_on_proc_frame:.4f}, {norm_y_on_proc_frame:.4f})")
            print(f"  Final pixel coordinates (preview on original): ({final_preview_x}, {final_preview_y})")
        
        print(f"[DEBUG] === END DEBUGGING ===")
        
        # Build output: single landmark and confidence
        # The landmarks returned should be normalized relative to the *processed_frame*
        # (the frame after FrameProcessor.preprocess but before letterboxing)
        # This is because main.py will scale these using processed_frame_shape and add crop_offsets.
        
        # Recalculate norm_x_on_proc_frame and norm_y_on_proc_frame for the return value
        # (This is a repeat of the logic in the debug print block, refactor if desired)
        model_h, model_w = self.model_loader.get_input_shape()
        proc_h, proc_w = self.processed_frame_shape
        scale_w_fit = model_w / proc_w
        scale_h_fit = model_h / proc_h
        scale_fit = min(scale_w_fit, scale_h_fit)
        new_proc_w_on_model = int(proc_w * scale_fit)
        new_proc_h_on_model = int(proc_h * scale_fit)
        pad_x_on_model = (model_w - new_proc_w_on_model) // 2
        pad_y_on_model = (model_h - new_proc_h_on_model) // 2
        abs_x_on_model = cx * model_w
        abs_y_on_model = cy * model_h
        x_on_resized_proc_frame = abs_x_on_model - pad_x_on_model
        y_on_resized_proc_frame = abs_y_on_model - pad_y_on_model
        
        # Ensure new_proc_w_on_model and new_proc_h_on_model are not zero before division
        norm_cx_for_main = (x_on_resized_proc_frame / new_proc_w_on_model) if new_proc_w_on_model > 0 else 0.0
        norm_cy_for_main = (y_on_resized_proc_frame / new_proc_h_on_model) if new_proc_h_on_model > 0 else 0.0
        
        norm_cx_for_main = np.clip(norm_cx_for_main, 0.0, 1.0)
        norm_cy_for_main = np.clip(norm_cy_for_main, 0.0, 1.0)

        landmarks = np.array([[[norm_cx_for_main, norm_cy_for_main, 0.0]]])
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
        """Convert coordinates from model space (normalized to model input dimensions,
           accounting for letterbox/pillarbox padding) to absolute pixel coordinates
           on the original, uncropped, unresized camera frame."""

        if self.original_frame_shape is None or self.processed_frame_shape is None:
            # Fallback if dimensions not set, though this shouldn't happen in normal flow
            return int(x * 640), int(y * 480) # Assuming a default, may not be accurate

        # Model output (x, y) is normalized to model_input_width, model_input_height
        model_input_height, model_input_width = self.model_loader.get_input_shape()
        
        # Dimensions of the frame that was letterboxed/pillarboxed (i.e., after FrameProcessor.preprocess)
        processed_h, processed_w = self.processed_frame_shape

        # Calculate the scaling factor and padding that were used in prepare_input
        scale_w = model_input_width / processed_w
        scale_h = model_input_height / processed_h
        scale = min(scale_w, scale_h)

        new_w_on_model_input = int(processed_w * scale) # width of processed_frame when scaled onto model_input
        new_h_on_model_input = int(processed_h * scale) # height of processed_frame when scaled onto model_input

        padding_x_on_model_input = (model_input_width - new_w_on_model_input) // 2
        padding_y_on_model_input = (model_input_height - new_h_on_model_input) // 2

        # 1. Denormalize (x, y) from model input dimensions to absolute pixel values on the model input tensor
        abs_x_on_model_input = x * model_input_width
        abs_y_on_model_input = y * model_input_height

        # 2. Remove padding to get coordinates relative to the scaled (but not padded) processed_frame
        x_on_scaled_processed_frame = abs_x_on_model_input - padding_x_on_model_input
        y_on_scaled_processed_frame = abs_y_on_model_input - padding_y_on_model_input

        # 3. Rescale these coordinates back to the dimensions of the processed_frame
        # (before it was scaled and padded for the model)
        # Avoid division by zero if new_w/h_on_model_input is zero
        x_on_processed_frame = (x_on_scaled_processed_frame / new_w_on_model_input) * processed_w if new_w_on_model_input > 0 else 0
        y_on_processed_frame = (y_on_scaled_processed_frame / new_h_on_model_input) * processed_h if new_h_on_model_input > 0 else 0
        
        # 4. Add crop offsets to convert to coordinates relative to the original camera frame
        # self.crop_offset_x and self.crop_offset_y are offsets in original frame pixels
        pixel_x_on_original_frame = x_on_processed_frame + self.crop_offset_x
        pixel_y_on_processed_frame = y_on_processed_frame + self.crop_offset_y

        # Clip to original frame boundaries
        original_h, original_w = self.original_frame_shape
        pixel_x_final = int(max(0, min(pixel_x_on_original_frame, original_w - 1)))
        pixel_y_final = int(max(0, min(pixel_y_on_processed_frame, original_h - 1)))

        return pixel_x_final, pixel_y_final
        
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
