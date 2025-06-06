"""
Improved confidence analyzer that uses multiple signals to determine hand presence.
"""

import numpy as np

class ConfidenceAnalyzer:
    """Analyzes multiple signals to determine if a hand is actually present."""
    
    def __init__(self, 
                 presence_threshold=0.3,  # Back to less strict
                 handedness_threshold=0.05,
                 landmark_consistency_threshold=0.4):  # Lower threshold
        """Initialize with thresholds for different confidence signals."""
        self.presence_threshold = presence_threshold
        self.handedness_threshold = handedness_threshold
        self.landmark_consistency_threshold = landmark_consistency_threshold
    
    def analyze_detection(self, landmarks, handedness, hand_presence_score, landmark_scores, frame_shape):
        """
        Analyze detection using multiple signals including landmark scores.
        
        Returns:
            tuple: (is_valid_detection, confidence_score, analysis_details)
        """
        analysis = {
            'presence_score': float(hand_presence_score[0][0]),
            'handedness_score': float(handedness[0][0]),
            'landmark_scores': landmark_scores[0] if landmark_scores is not None else None,
            'landmark_consistency': 0.0,
            'position_validity': False,
            'overall_confidence': 0.0
        }
        
        # 1. Check basic presence score
        presence_ok = analysis['presence_score'] >= self.presence_threshold
        
        # 2. Check handedness confidence (higher values suggest more confident detection)
        handedness_ok = analysis['handedness_score'] >= self.handedness_threshold
        
        # 3. Analyze landmark consistency
        if landmarks is not None and len(landmarks) > 0:
            analysis['landmark_consistency'] = self._analyze_landmark_consistency(landmarks)
            analysis['position_validity'] = self._check_position_validity(landmarks, frame_shape)
        
        # 4. Analyze landmark scores (NEW: use the 4th model output)
        landmark_scores_ok = True
        if analysis['landmark_scores'] is not None:
            landmark_scores_ok = self._analyze_landmark_scores(analysis['landmark_scores'])
            analysis['landmark_scores_valid'] = landmark_scores_ok
        
        # 5. Calculate overall confidence
        landmark_ok = analysis['landmark_consistency'] >= self.landmark_consistency_threshold
        
        # Combined decision logic - now includes landmark scores
        signals_passed = sum([presence_ok, handedness_ok, landmark_ok, analysis['position_validity'], landmark_scores_ok])
        
        # More strict validation: require landmark scores to be valid for positive detection
        # Also increase the threshold requirements
        is_valid = (presence_ok and landmark_scores_ok and signals_passed >= 4)
        
        # Calculate weighted confidence score (including landmark scores)
        analysis['overall_confidence'] = (
            analysis['presence_score'] * 0.3 +
            analysis['handedness_score'] * 0.15 +
            analysis['landmark_consistency'] * 0.25 +
            (1.0 if analysis['position_validity'] else 0.0) * 0.1 +
            (1.0 if landmark_scores_ok else 0.0) * 0.2  # NEW: landmark scores weight
        )
        
        return is_valid, analysis['overall_confidence'], analysis
    
    def _analyze_landmark_consistency(self, landmarks):
        """Check if landmarks form a consistent hand shape."""
        try:
            # Reshape landmarks to [21, 3] if needed
            if landmarks.ndim == 1 and len(landmarks) == 63:
                hand_landmarks = landmarks.reshape(21, 3)
            elif landmarks.ndim == 2 and landmarks.shape[0] == 21:
                hand_landmarks = landmarks
            else:
                return 0.0
            
            # Check basic anatomical constraints
            
            # 1. Wrist should be roughly in the center of finger bases
            wrist = hand_landmarks[0]  # Landmark 0 is wrist
            finger_bases = hand_landmarks[[5, 9, 13, 17]]  # Finger base landmarks
            
            # Calculate average finger base position
            avg_finger_base = np.mean(finger_bases, axis=0)
            
            # Distance from wrist to average finger base should be reasonable (not too far)
            wrist_to_fingers_dist = np.linalg.norm(wrist[:2] - avg_finger_base[:2])
            distance_score = 1.0 / (1.0 + wrist_to_fingers_dist / 30.0)  # Penalize large distances
            
            # 2. Check finger progression (each finger should get progressively further from wrist)
            finger_progression_score = 0.0
            for finger_start in [1, 5, 9, 13, 17]:  # Start of each finger
                finger_joints = hand_landmarks[finger_start:finger_start+4]
                if len(finger_joints) == 4:
                    # Calculate distances from wrist for each joint
                    distances = [np.linalg.norm(joint[:2] - wrist[:2]) for joint in finger_joints]
                    # Check if distances generally increase (finger extends from hand)
                    progression_ok = sum([distances[i] <= distances[i+1] for i in range(3)]) >= 2
                    finger_progression_score += 1.0 if progression_ok else 0.0
            finger_progression_score /= 5.0  # Normalize by number of fingers
            
            # 3. Check if landmarks are clustered reasonably (not spread all over)
            landmark_spread = np.std(hand_landmarks[:, :2], axis=0)
            spread_score = 1.0 / (1.0 + np.mean(landmark_spread) / 40.0)  # More strict spread
            
            # 4. Check Z-depth consistency (Z values should be relatively similar)
            z_values = hand_landmarks[:, 2]
            z_consistency = 1.0 / (1.0 + np.std(z_values) / 10.0)  # More strict z consistency
            
            # 5. Check finger tip positions relative to bases
            finger_tip_score = 0.0
            for i, (base_idx, tip_idx) in enumerate([(5, 8), (9, 12), (13, 16), (17, 20)]):  # Skip thumb for now
                base_pos = hand_landmarks[base_idx][:2]
                tip_pos = hand_landmarks[tip_idx][:2]
                tip_distance = np.linalg.norm(tip_pos - base_pos)
                # Finger length should be reasonable (not too short or too long)
                if 20 < tip_distance < 80:  # Reasonable finger length in pixels
                    finger_tip_score += 1.0
            finger_tip_score /= 4.0  # 4 fingers checked
            
            # Combine consistency measures with weights
            consistency_score = (
                distance_score * 0.25 +
                finger_progression_score * 0.25 +
                spread_score * 0.2 +
                z_consistency * 0.1 +
                finger_tip_score * 0.2
            )
            
            return min(1.0, max(0.0, consistency_score))
            
        except Exception as e:
            return 0.0
    
    def _check_position_validity(self, landmarks, frame_shape):
        """Check if landmark positions are within reasonable bounds."""
        try:
            if landmarks.ndim == 1 and len(landmarks) == 63:
                hand_landmarks = landmarks.reshape(21, 3)
            elif landmarks.ndim == 2 and landmarks.shape[0] == 21:
                hand_landmarks = landmarks
            else:
                return False
            
            img_height, img_width = frame_shape[:2]
            
            # Check if most landmarks are within frame bounds (allowing some margin)
            margin = 20
            x_coords = hand_landmarks[:, 0]
            y_coords = hand_landmarks[:, 1]
            
            valid_x = np.sum((x_coords >= -margin) & (x_coords <= img_width + margin))
            valid_y = np.sum((y_coords >= -margin) & (y_coords <= img_height + margin))
            
            # At least 15 out of 21 landmarks should be in valid positions
            return (valid_x >= 15) and (valid_y >= 15)
            
        except Exception as e:
            return False
    
    def _analyze_landmark_scores(self, landmark_scores):
        """
        Analyze the landmark confidence scores from the 4th model output.
        
        Args:
            landmark_scores: Array of 63 values (21 landmarks × 3 scores each)
        
        Returns:
            bool: True if landmark scores indicate a valid hand detection
        """
        try:
            if landmark_scores is None or len(landmark_scores) != 63:
                return False
            
            # Reshape to [21, 3] if it's flattened
            if landmark_scores.ndim == 1:
                scores = landmark_scores.reshape(21, 3)
            else:
                scores = landmark_scores
            
            # Analyze score patterns that indicate real hands vs false positives
            # Real hands should have:
            # 1. Most scores should be positive (indicating visible landmarks)
            # 2. Scores should not be uniformly distributed
            # 3. Key landmarks (wrist, fingertips) should have higher confidence
            
            # Count positive vs negative scores
            positive_scores = np.sum(scores > 0)
            total_scores = scores.size
            positive_ratio = positive_scores / total_scores
            
            # Check score variance (false positives tend to have low variance)
            score_variance = np.var(scores)
            
            # Check key landmark scores (wrist=0, fingertips=4,8,12,16,20)
            key_landmarks = [0, 4, 8, 12, 16, 20]
            key_scores = scores[key_landmarks].flatten()
            key_positive_ratio = np.sum(key_scores > 0) / len(key_scores)
            
            # Decision criteria based on observations from debug output
            # Gray background had mixed positive/negative with low variance
            # Real hands should have more consistent positive scores for key landmarks
            
            criteria = [
                positive_ratio > 0.6,  # At least 60% positive scores
                score_variance > 0.0001,  # Some variance in scores
                key_positive_ratio > 0.7,  # Key landmarks mostly positive
                np.mean(np.abs(scores)) > 0.005  # Average absolute score above threshold
            ]
            
            # Require at least 3 out of 4 criteria
            return sum(criteria) >= 3
            
        except Exception as e:
            print(f"[ERROR] Landmark score analysis failed: {e}")
            return False
