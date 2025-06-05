#!/usr/bin/env python3
"""
Download and prepare TensorFlow Lite hand landmark model for testing
"""
import os
import sys
import urllib.request
import urllib.error

def download_model():
    """Download a compatible hand landmark model"""
    
    model_urls = [
        # MediaPipe hand landmark model (this should work with our inference pipeline)
        {
            'url': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
            'filename': 'hand_landmarker.task',
            'description': 'MediaPipe Hand Landmarker (Task format)'
        },
        # Alternative lightweight hand detection model
        {
            'url': 'https://tfhub.dev/mediapipe/lite-model/hands/1?lite-format=tflite',
            'filename': 'hand_detection_lite.tflite', 
            'description': 'MediaPipe Hand Detection (TFLite)'
        }
    ]
    
    print("Available hand landmark models:")
    for i, model in enumerate(model_urls):
        print(f"{i+1}. {model['description']}")
        print(f"   URL: {model['url']}")
        print(f"   File: {model['filename']}")
        print()
    
    # Try to download the first model
    model = model_urls[0]
    print(f"Attempting to download: {model['description']}")
    
    try:
        print(f"Downloading {model['filename']}...")
        urllib.request.urlretrieve(model['url'], model['filename'])
        
        if os.path.exists(model['filename']):
            size = os.path.getsize(model['filename'])
            print(f"✓ Downloaded {model['filename']} ({size} bytes)")
            return model['filename']
        else:
            print("✗ Download failed - file not found")
            return None
            
    except urllib.error.URLError as e:
        print(f"✗ Download failed: {e}")
        return None
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return None

def create_dummy_model():
    """Create a simple dummy TFLite model for testing the pipeline"""
    print("Creating dummy model for pipeline testing...")
    
    try:
        import tensorflow as tf
        
        # Create a simple model that outputs random landmarks
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(63)  # 21 landmarks * 3 coordinates
        ])
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the model
        with open('dummy_hand_model.tflite', 'wb') as f:
            f.write(tflite_model)
            
        print("✓ Created dummy_hand_model.tflite for testing")
        return 'dummy_hand_model.tflite'
        
    except ImportError:
        print("✗ TensorFlow not available for creating dummy model")
        return None
    except Exception as e:
        print(f"✗ Failed to create dummy model: {e}")
        return None

def main():
    print("=== Hand Tracking Model Setup ===\n")
    
    # Check current directory
    print(f"Working directory: {os.getcwd()}")
    
    # Try to download a real model first
    model_file = download_model()
    
    if model_file is None:
        print("\nReal model download failed. Creating dummy model for testing...")
        model_file = create_dummy_model()
    
    if model_file:
        print(f"\n✓ Model ready: {model_file}")
        print(f"  Use with: python capture_hand_tracking.py --model {model_file}")
        
        # Test if our script can load it
        print(f"\nTesting model loading...")
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite import Interpreter
            except ImportError:
                print("✗ No TFLite interpreter available")
                return
        
        try:
            interpreter = Interpreter(model_path=model_file)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"✓ Model loads successfully")
            print(f"  Input shape: {input_details[0]['shape']}")
            print(f"  Output shape: {output_details[0]['shape']}")
            
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
    else:
        print("\n✗ No model available for testing")
        print("\nManual download options:")
        print("1. Download from TensorFlow Hub: https://tfhub.dev/mediapipe/")
        print("2. Use MediaPipe models: https://developers.google.com/mediapipe/")
        print("3. Create your own with TensorFlow Lite Model Maker")

if __name__ == '__main__':
    main()
