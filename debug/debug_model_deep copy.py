#!/usr/bin/env python3
"""
Deep debug of model behavior and input/output analysis.
"""

import cv2
import numpy as np
import tensorflow as tf

# Import modules
from detection import ModelLoader, InferenceEngine
from camera import FrameProcessor


def analyze_model_details():
    """Analyze the model's detailed specifications."""
    print("[DEBUG] === MODEL ANALYSIS ===")
    
    model_path = "/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite"
    
    # Load model directly with TensorFlow Lite
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"[MODEL] Model path: {model_path}")
    print(f"[MODEL] Input details:")
    for i, detail in enumerate(input_details):
        print(f"  Input {i}: {detail}")
    
    print(f"[MODEL] Output details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail}")
    
    return interpreter, input_details, output_details


def test_input_formats():
    """Test different input formats to see what the model expects."""
    print(f"\n[DEBUG] === INPUT FORMAT TESTING ===")
    
    interpreter, input_details, output_details = analyze_model_details()
    
    # Test different input formats
    input_shape = input_details[0]['shape']
    height, width = input_shape[1], input_shape[2]
    print(f"[INPUT] Expected shape: {input_shape}")
    
    # Test cases
    test_cases = [
        ("All zeros", np.zeros((1, height, width, 3), dtype=np.float32)),
        ("All ones", np.ones((1, height, width, 3), dtype=np.float32)),
        ("Random noise", np.random.random((1, height, width, 3)).astype(np.float32)),
        ("Gray (0.5)", np.full((1, height, width, 3), 0.5, dtype=np.float32)),
        ("Black RGB", np.zeros((1, height, width, 3), dtype=np.float32)),
        ("White RGB", np.ones((1, height, width, 3), dtype=np.float32)),
    ]
    
    for case_name, input_tensor in test_cases:
        print(f"\n[TEST] {case_name}:")
        print(f"  Input shape: {input_tensor.shape}")
        print(f"  Input range: {input_tensor.min():.3f} to {input_tensor.max():.3f}")
        print(f"  Input mean: {input_tensor.mean():.3f}")
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        
        # Run inference
        interpreter.invoke()
        
        # Get outputs
        landmarks = interpreter.get_tensor(output_details[0]['index'])
        handedness = interpreter.get_tensor(output_details[1]['index']) 
        hand_scores = interpreter.get_tensor(output_details[2]['index'])
        
        print(f"  Output landmarks shape: {landmarks.shape}")
        print(f"  Output handedness: {handedness}")
        print(f"  Output hand_scores: {hand_scores}")
        print(f"  Presence score: {float(hand_scores[0][0]):.6f}")


def test_preprocessing_pipeline():
    """Test our preprocessing pipeline vs direct model input."""
    print(f"\n[DEBUG] === PREPROCESSING PIPELINE TEST ===")
    
    # Initialize our pipeline
    model_loader = ModelLoader(model_path="/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite")
    if not model_loader.load_model():
        print("[ERROR] Failed to load model")
        return
        
    inference_engine = InferenceEngine(model_loader)
    frame_processor = FrameProcessor(crop_factor=0.8)
    
    # Create test image - solid gray
    width, height = 640, 480
    test_frame = np.full((height, width, 3), 128, dtype=np.uint8)
    
    print(f"[ORIGINAL] Frame shape: {test_frame.shape}")
    print(f"[ORIGINAL] Frame dtype: {test_frame.dtype}")
    print(f"[ORIGINAL] Frame range: {test_frame.min()} to {test_frame.max()}")
    print(f"[ORIGINAL] Frame mean: {test_frame.mean():.1f}")
    
    # Step 1: Frame processor preprocessing
    processed_frame = frame_processor.preprocess(test_frame)
    print(f"[PROCESSED] Frame shape: {processed_frame.shape}")
    print(f"[PROCESSED] Frame dtype: {processed_frame.dtype}")
    print(f"[PROCESSED] Frame range: {processed_frame.min()} to {processed_frame.max()}")
    print(f"[PROCESSED] Frame mean: {processed_frame.mean():.1f}")
    
    # Step 2: Inference engine input preparation
    input_tensor = inference_engine.prepare_input(processed_frame)
    print(f"[INPUT_TENSOR] Shape: {input_tensor.shape}")
    print(f"[INPUT_TENSOR] Dtype: {input_tensor.dtype}")
    print(f"[INPUT_TENSOR] Range: {input_tensor.min():.6f} to {input_tensor.max():.6f}")
    print(f"[INPUT_TENSOR] Mean: {input_tensor.mean():.6f}")
    
    # Step 3: Run inference using our pipeline
    landmarks, handedness, hand_scores = inference_engine.run_inference(input_tensor)
    print(f"[PIPELINE_RESULT] Hand scores: {hand_scores}")
    print(f"[PIPELINE_RESULT] Presence: {float(hand_scores[0][0]):.6f}")
    
    # Step 4: Compare with direct model input (no preprocessing)
    print(f"\n[COMPARISON] Testing direct model input...")
    
    # Create direct input - gray image normalized
    direct_input = np.full((1, 224, 224, 3), 128.0/255.0, dtype=np.float32)
    print(f"[DIRECT_INPUT] Shape: {direct_input.shape}")
    print(f"[DIRECT_INPUT] Range: {direct_input.min():.6f} to {direct_input.max():.6f}")
    print(f"[DIRECT_INPUT] Mean: {direct_input.mean():.6f}")
    
    # Run inference directly
    interpreter = model_loader.interpreter
    interpreter.set_tensor(model_loader.input_details[0]['index'], direct_input)
    interpreter.invoke()
    
    direct_landmarks = interpreter.get_tensor(model_loader.output_details[0]['index'])
    direct_handedness = interpreter.get_tensor(model_loader.output_details[1]['index'])
    direct_hand_scores = interpreter.get_tensor(model_loader.output_details[2]['index'])
    
    print(f"[DIRECT_RESULT] Hand scores: {direct_hand_scores}")
    print(f"[DIRECT_RESULT] Presence: {float(direct_hand_scores[0][0]):.6f}")
    
    # Compare results
    pipeline_presence = float(hand_scores[0][0])
    direct_presence = float(direct_hand_scores[0][0])
    
    print(f"\n[COMPARISON_RESULT]")
    print(f"  Pipeline presence: {pipeline_presence:.6f}")
    print(f"  Direct presence: {direct_presence:.6f}")
    print(f"  Difference: {abs(pipeline_presence - direct_presence):.6f}")
    
    if abs(pipeline_presence - direct_presence) > 0.001:
        print(f"  ⚠️  SIGNIFICANT DIFFERENCE - preprocessing might be the issue!")
    else:
        print(f"  ✅ Similar results - preprocessing is not the issue")


def test_mediapipe_official():
    """Test what happens with MediaPipe's official interface."""
    print(f"\n[DEBUG] === MEDIAPIPE OFFICIAL COMPARISON ===")
    
    try:
        import mediapipe as mp
        
        # Initialize MediaPipe hands
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.1,  # Very low threshold
            min_tracking_confidence=0.1
        )
        
        # Create test image - solid gray
        width, height = 640, 480
        test_frame = np.full((height, width, 3), 128, dtype=np.uint8)
        
        # Convert BGR to RGB (MediaPipe expects RGB)
        rgb_frame = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        print(f"[MEDIAPIPE] Results: {results}")
        print(f"[MEDIAPIPE] Multi hand landmarks: {results.multi_hand_landmarks}")
        print(f"[MEDIAPIPE] Multi handedness: {results.multi_handedness}")
        
        if results.multi_hand_landmarks:
            print(f"[MEDIAPIPE] ⚠️  FALSE POSITIVE: MediaPipe also detects hand on gray background!")
        else:
            print(f"[MEDIAPIPE] ✅ Correct: MediaPipe correctly rejects gray background")
            
        hands.close()
        
    except ImportError:
        print(f"[MEDIAPIPE] MediaPipe not available for comparison")


if __name__ == "__main__":
    analyze_model_details()
    test_input_formats()
    test_preprocessing_pipeline()
    test_mediapipe_official()
