# Real-Time Hand Tracking on Raspberry Pi 5

A robust Python script for real-time hand tracking using Raspberry Pi 5 camera and TensorFlow Lite model inference.

## Features

✅ **Multiple Camera Backends**
- Primary: `libcamera` subprocess method (optimized for Pi 5)
- Fallback: `v4l2` direct access
- Auto-detection of working camera method

✅ **Optimized Performance**
- Multi-threaded frame capture with background processing
- Frame buffering optimized for real-time performance (single buffer)
- FIFO-based streaming with aggressive frame dropping
- libcamera configured with --nopreview for maximum performance
- **Performance**: ~30-35 FPS with 15-17ms inference time
- Minimal latency buffering and optimized YUV-to-BGR conversion

✅ **Hand Landmark Detection**
- 21-point hand landmark detection using TensorFlow Lite
- Wrist visualization (red circle) with optional all-landmark display
- Normalized coordinate output (0-1 range)
- Real-time coordinate tracking

✅ **Robust Error Handling**
- Comprehensive camera detection and troubleshooting
- Graceful fallback between camera methods
- Resource cleanup and process management

✅ **Flexible Operation Modes**
- GUI mode with live video display
- Headless mode for remote/server deployment
- Debug mode with detailed coordinate output
- Configurable resolution and frame rates

## Installation

### Prerequisites

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required system packages
sudo apt install -y python3-pip python3-venv libcamera-apps

# Install OpenCV dependencies
sudo apt install -y libopencv-dev python3-opencv

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options -> Camera -> Enable
```

### Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install opencv-python numpy tflite-runtime
# OR if using full TensorFlow:
# pip install opencv-python numpy tensorflow
```

### Model Setup

The script uses `hand_landmark_lite.tflite` model (already included).

## Usage

### Basic Usage

```bash
# Auto-detect camera and run with GUI
python3 capture_hand_tracking.py

# Specify camera backend
python3 capture_hand_tracking.py --backend libcamera

# Run in headless mode (no GUI)
python3 capture_hand_tracking.py --headless

# Enable debug output
python3 capture_hand_tracking.py --debug

# Draw all 21 landmarks (not just wrist)
python3 capture_hand_tracking.py --draw_all
```

### Advanced Options

```bash
# Custom resolution
python3 capture_hand_tracking.py --res 320x240

# Use custom model
python3 capture_hand_tracking.py --model your_model.tflite

# Test camera detection
python3 capture_hand_tracking.py --test_camera

# Run troubleshooting
python3 capture_hand_tracking.py --troubleshoot
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `hand_landmark_lite.tflite` | Path to TFLite model |
| `--res` | `640x480` | Camera resolution (WxH) |
| `--backend` | `auto` | Camera backend (`libcamera`, `v4l2`, `auto`) |
| `--draw_all` | `False` | Draw all 21 landmarks |
| `--headless` | `False` | Run without GUI |
| `--debug` | `False` | Enable debug output |
| `--test_camera` | `False` | Test camera and exit |
| `--troubleshoot` | `False` | Run diagnostics |

## Performance Metrics

**Raspberry Pi 5 Performance (Optimized):**
- **FPS**: 30-35 FPS sustained (peak performance)
- **Inference Time**: 15-17ms per frame (optimized)
- **Resolution**: 640x480 (configurable down to 320x240)
- **CPU Usage**: ~40-60% single core
- **Memory**: ~200MB RAM

**Performance Optimizations:**
- `--nopreview` flag eliminates libcamera preview overhead
- Single-frame buffer queue for minimal latency
- Aggressive frame dropping for real-time performance
- Optimized YUV420 to BGR conversion
- Reduced FIFO timeouts and buffer counts

## Architecture

### LibcameraCapture Class (Primary - Optimized)
- Uses `libcamera-vid` with `--nopreview` for maximum performance
- Multi-threaded frame reading with background processing
- FIFO pipe with minimal buffering for lowest latency
- Single-frame queue with aggressive old-frame dropping
- Optimized YUV420 to BGR conversion with reduced timeout

### TensorFlow Lite Inference
- Optimized XNNPACK delegate for CPU acceleration
- Input preprocessing: RGB conversion and normalization
- Output post-processing: coordinate normalization and validation
- Error handling for various model output formats

### Visualization Pipeline
- Real-time landmark drawing on video frames
- Coordinate clamping to prevent out-of-bounds drawing
- FPS and inference time overlay
- Support for headless operation

## Troubleshooting

### Camera Issues

```bash
# Run diagnostics
python3 capture_hand_tracking.py --troubleshoot

# Test camera detection
python3 capture_hand_tracking.py --test_camera

# Check libcamera functionality
libcamera-hello --list-cameras
libcamera-still -o test.jpg
```

### Common Solutions

1. **Camera not detected**:
   ```bash
   sudo raspi-config  # Enable camera
   sudo reboot
   ```

2. **Permission issues**:
   ```bash
   sudo usermod -a -G video $USER
   # Logout and login again
   ```

3. **Qt display errors** (on headless systems):
   ```bash
   # Use headless mode
   python3 capture_hand_tracking.py --headless
   ```

4. **Low performance**:
   ```bash
   # Reduce resolution
   python3 capture_hand_tracking.py --res 320x240
   ```

## Code Structure

```
capture_hand_tracking.py
├── Argument parsing and configuration
├── LibcameraCapture class (optimized camera access)
├── Camera detection and testing functions
├── TensorFlow Lite model loading and inference
├── Hand landmark visualization
├── Main processing loop with error handling
└── Comprehensive troubleshooting system
```

## Key Improvements Made

1. **Removed PiCamera2 dependency** - Focused solely on libcamera implementation
2. **Optimized LibcameraCapture** - Added multi-threading and frame buffering
3. **Fixed coordinate normalization** - Proper handling of landmark coordinates
4. **Added headless mode** - Support for systems without display
5. **Enhanced error handling** - Robust resource cleanup and error recovery
6. **Performance monitoring** - Real-time FPS and inference time tracking

## Development Notes

- The system prioritizes `libcamera` backend as it's the modern standard for Pi 5
- Threading is used to separate frame capture from processing for better performance
- Coordinate normalization handles both pre-normalized and raw model outputs
- Resource cleanup ensures proper process termination and temp file removal

## License

This project is provided as-is for educational and development purposes.
