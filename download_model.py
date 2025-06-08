#!/usr/bin/env python3
"""
Script to download the MediaPipe palm detection TensorFlow Lite model
and update project paths to use the local model.
"""

import os
import urllib.request
import hashlib
import sys
import zipfile
import tempfile
from pathlib import Path

# Model information - Using palm detection model for faster FPS
# Palm detection is optimized for speed rather than accuracy of finger positions
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/palm_detection/palm_detection/float16/latest/palm_detection.task"
MODEL_FILENAME = "palm_detection.tflite"
MODELS_DIR = Path(__file__).parent / "models"

# Expected SHA256 hash of the model (for verification)
# Skip hash verification for the palm detection model to allow for updates
EXPECTED_HASH = None

def download_model():
    """Download the MediaPipe palm detection model."""
    print("[INFO] Downloading MediaPipe palm detection model...")
    
    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(exist_ok=True)
    
    model_path = MODELS_DIR / MODEL_FILENAME
    
    # Check if model already exists
    if model_path.exists():
        print(f"[INFO] Model already exists at: {model_path}")
        return str(model_path)
    
    try:
        # Download the model bundle to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.task', delete=False) as temp_file:
            print(f"[INFO] Downloading from: {MODEL_URL}")
            urllib.request.urlretrieve(MODEL_URL, temp_file.name)
            
            # Extract the TFLite model from the bundle
            print("[INFO] Extracting TFLite model from bundle...")
            
            with zipfile.ZipFile(temp_file.name, 'r') as zip_ref:
                # List contents to see what's available
                file_list = zip_ref.namelist()
                print(f"[INFO] Bundle contents: {file_list}")
                
                # Look for palm detection model
                tflite_candidates = [
                    'palm_detection.tflite',
                    'palm_detector.tflite', 
                    'hand_detector.tflite',
                    'detector.tflite'
                ]
                
                extracted_file = None
                for candidate in tflite_candidates:
                    if candidate in file_list:
                        print(f"[INFO] Found model: {candidate}")
                        with zip_ref.open(candidate) as source:
                            with open(model_path, 'wb') as target:
                                target.write(source.read())
                        extracted_file = candidate
                        break
                
                if not extracted_file:
                    # If no exact match, try the first .tflite file
                    tflite_files = [f for f in file_list if f.endswith('.tflite')]
                    if tflite_files:
                        extracted_file = tflite_files[0]
                        print(f"[INFO] Using first TFLite file found: {extracted_file}")
                        with zip_ref.open(extracted_file) as source:
                            with open(model_path, 'wb') as target:
                                target.write(source.read())
                    else:
                        print("[ERROR] No TFLite files found in bundle")
                        return None
            
            # Clean up temp file
            os.unlink(temp_file.name)
        
        print(f"[SUCCESS] Model extracted to: {model_path}")
        
        # Verify file size
        file_size = model_path.stat().st_size
        print(f"[INFO] Extracted file size: {file_size:,} bytes")
        
        if file_size < 1000:  # Model should be much larger
            print("[WARNING] Extracted file seems too small, might be an error")
            return None
            
        return str(model_path)
        
    except Exception as e:
        print(f"[ERROR] Failed to download/extract model: {e}")
        
        # Try alternative: extract from installed MediaPipe package
        print("[INFO] Trying to copy from installed MediaPipe package...")
        return copy_from_mediapipe()

def copy_from_mediapipe():
    """Copy model from installed MediaPipe package as fallback."""
    # Look for palm detection models in MediaPipe installation
    mediapipe_paths = [
        "venv/lib/python3.11/site-packages/mediapipe/modules/palm_detection/palm_detection_lite.tflite",
        "venv/lib64/python3.11/site-packages/mediapipe/modules/palm_detection/palm_detection_lite.tflite",
        "/usr/local/lib/python3.11/site-packages/mediapipe/modules/palm_detection/palm_detection_lite.tflite",
        # Fallback to hand landmark models if palm detection not found
        "venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite",
        "venv/lib64/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite",
        "/usr/local/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite"
    ]
    
    project_root = Path(__file__).parent
    
    for relative_path in mediapipe_paths:
        source_path = project_root / relative_path
        if source_path.exists():
            target_path = MODELS_DIR / MODEL_FILENAME
            
            try:
                import shutil
                shutil.copy2(source_path, target_path)
                print(f"[SUCCESS] Copied model from: {source_path}")
                print(f"[SUCCESS] Model saved to: {target_path}")
                return str(target_path)
            except Exception as e:
                print(f"[ERROR] Failed to copy from {source_path}: {e}")
                continue
    
    print("[ERROR] Could not find MediaPipe model in any expected location")
    return None

def update_config_file():
    """Update the settings.py file to use the local model."""
    config_file = Path(__file__).parent / "config" / "settings.py"
    
    if not config_file.exists():
        print(f"[ERROR] Config file not found: {config_file}")
        return False
    
    try:
        # Read current config
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Update the model path to use the palm detection model
        # Handle both old and new formats
        old_patterns = [
            'DEFAULT_MODEL_PATH = "hand_landmark_lite.tflite"',
            'DEFAULT_MODEL_PATH = "models/hand_landmark_lite.tflite"',
            'DEFAULT_MODEL_PATH = "models/hand_landmark_full.tflite"'
        ]
        new_line = 'DEFAULT_MODEL_PATH = "models/palm_detection.tflite"'
        
        updated = False
        for old_pattern in old_patterns:
            if old_pattern in content:
                updated_content = content.replace(old_pattern, new_line)
                content = updated_content
                updated = True
                break
            
        if updated:
            # Write back
            with open(config_file, 'w') as f:
                f.write(content)
            
            print(f"[SUCCESS] Updated config file: {config_file}")
            return True
        else:
            print(f"[WARNING] Could not find expected line in config file")
            print(f"[INFO] Please manually update DEFAULT_MODEL_PATH to: models/palm_detection.tflite")
            return False
            
    except Exception as e:
        print(f"[ERROR] Failed to update config file: {e}")
        return False

def update_debug_scripts():
    """Update debug scripts to use the new model path."""
    debug_dir = Path(__file__).parent / "debug"
    scripts_to_update = [
        "debug_simple_bg.py",
        "debug_all_outputs.py", 
        "debug_landmark_scores.py"
    ]
    
    old_paths = [
        "/home/rtsvetkov/hand_tracking/venv/lib/python3.11/site-packages/mediapipe/modules/hand_landmark/hand_landmark_lite.tflite",
        "/home/rtsvetkov/hand_tracking/models/hand_landmark_full.tflite",
        "/home/rtsvetkov/hand_tracking/models/hand_landmark_lite.tflite"
    ]
    new_path = "/home/rtsvetkov/hand_tracking/models/palm_detection.tflite"
    
    for script_name in scripts_to_update:
        script_path = debug_dir / script_name
        
        # Also check in root directory
        if not script_path.exists():
            script_path = Path(__file__).parent / script_name
        
        if script_path.exists():
            try:
                with open(script_path, 'r') as f:
                    content = f.read()
                
                updated = False
                for old_path in old_paths:
                    if old_path in content:
                        updated_content = content.replace(old_path, new_path)
                        content = updated_content
                        updated = True
                
                if updated:
                    with open(script_path, 'w') as f:
                        f.write(content)
                    print(f"[SUCCESS] Updated: {script_path}")
                else:
                    print(f"[INFO] No update needed for: {script_path}")
                    
            except Exception as e:
                print(f"[ERROR] Failed to update {script_path}: {e}")
        else:
            print(f"[WARNING] Script not found: {script_path}")

def main():
    """Main function to download model and update paths."""
    print("=" * 60)
    print("MediaPipe Palm Detection Model Setup")
    print("=" * 60)
    
    # Download the model
    model_path = download_model()
    
    if not model_path:
        print("[ERROR] Failed to download or copy model")
        sys.exit(1)
    
    # Verify the model file
    if not Path(model_path).exists():
        print(f"[ERROR] Model file not found after download: {model_path}")
        sys.exit(1)
    
    file_size = Path(model_path).stat().st_size
    print(f"[INFO] Model file size: {file_size:,} bytes")
    
    # Update configuration
    print("\n[INFO] Updating project configuration...")
    update_config_file()
    
    # Update debug scripts
    print("\n[INFO] Updating debug scripts...")
    update_debug_scripts()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"Model location: {model_path}")
    print("Configuration updated to use local palm detection model.")
    print("You can now run the hand tracking application with faster palm detection.")
    print("=" * 60)

if __name__ == "__main__":
    main()
