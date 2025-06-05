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

### Automated Installation

```bash
# Make the installation script executable
chmod +x install.sh

# Run the installation script
./install.sh
```

The installation script will:
1. Update system packages
2. Install required system dependencies
3. Create a Python virtual environment
4. Install Python dependencies
5. Enable the camera interface (if needed)

### Manual Installation

If you prefer to install manually, follow these steps:

#### Prerequisites

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

#### Python Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Model Setup

The script uses `hand_landmark_lite.tflite` model (already included).

### Testing Your Setup

To verify that everything is working correctly, run the test script:

```bash
# Activate the virtual environment first
source venv/bin/activate

# Run the test script
python test_setup.py
```

This will check:
1. Required Python dependencies
2. Model loading
3. Camera functionality

The test will generate a `test_frame.jpg` file if the camera is working correctly.

## Usage

### Basic Usage

```bash
# Run with specified resolution (required)
python3 main.py --res 640x480

# Run in headless mode (no GUI)
python3 main.py --res 640x480 --headless

# Enable debug output
python3 main.py --res 640x480 --debug
```

### Advanced Options

```bash
# Custom resolution
python3 main.py --res 320x240

# Use custom model
python3 main.py --model custom_model.tflite

# Adjust confidence threshold
python3 main.py --res 640x480 --confidence 0.7

# Adjust coordinate smoothing
python3 main.py --res 640x480 --smoothing 0.6

# Legacy script (for reference)
python3 capture_hand_tracking_simple.py --res 640x480

# Test camera detection
python3 capture_hand_tracking.py --test_camera

# Run troubleshooting
python3 capture_hand_tracking.py --troubleshoot
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `hand_landmark_lite.tflite` | Path to TFLite model |
| `--res` | Required | Camera resolution (WxH), e.g. 640x480 |
| `--fps` | `30` | Camera frame rate |
| `--frame_skip` | `1` | Process every Nth frame |
| `--headless` | `False` | Run without GUI |
| `--debug` | `False` | Enable debug output |
| `--confidence` | `0.6` | Minimum confidence threshold (0-1) |
| `--smoothing` | `0.4` | Coordinate smoothing factor (0-1) |
| `--crop_factor` | `0.8` | Center crop factor for better accuracy (0-1) |
| `--max_jump` | `150` | Maximum pixel jump to filter noise |
| `--detection_loss_frames` | `5` | Frames before declaring hand lost |
| `--stability_threshold` | `0.6` | Minimum tracking quality for stable detection |
| `--false_positive_threshold` | `5` | Frames of suspicious detection before rejection |

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
# Check libcamera functionality
libcamera-hello --list-cameras
libcamera-still -o test.jpg

# Test camera directly
v4l2-ctl --list-devices
v4l2-ctl --all -d /dev/video0
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
   python3 main.py --res 640x480 --headless
   ```

4. **Low performance**:
   ```bash
   # Reduce resolution
   python3 main.py --res 320x240
   
   # Skip frames for better performance
   python3 main.py --res 640x480 --frame_skip 2
   ```

5. **Qt platform plugin warning**:
   ```
   qt.qpa.plugin: Could not find the Qt platform plugin "wayland" in "..."
   ```
   This is a harmless warning related to OpenCV's GUI components and doesn't affect functionality.
   
6. **Model output format errors**:
   If you get errors related to landmark processing, try using a different model or check that your
   model outputs hand landmarks in the expected format (21 points with x,y,z coordinates).

## Code Structure

For a detailed overview of the system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

### Modular Architecture

```
main.py                        # Main entry point and application orchestration
├── config/                    # Configuration management
│   └── settings.py            # Application-wide settings
├── camera/                    # Camera handling
│   ├── libcamera_capture.py   # LibCamera interface
│   └── frame_processor.py     # Frame preprocessing
├── detection/                 # ML model detection
│   ├── model_loader.py        # TFLite model loading
│   ├── inference_engine.py    # Model inference
│   └── landmark_extractor.py  # Landmark processing
├── tracking/                  # Position tracking
│   ├── hand_tracker.py        # Hand tracking logic
│   ├── position_validator.py  # Position validation
│   └── coordinate_smoother.py # Position smoothing
├── visualization/             # Display and UI
│   ├── renderer.py            # Drawing functions
│   ├── status_display.py      # Status information
│   └── frame_renderer.py      # Frame rendering
└── utils/                     # Utility functions
    ├── arg_parser.py          # Command line parsing
    └── timing_utils.py        # Performance monitoring
```

### Legacy Scripts (for reference)
```
archive/                         # Directory containing original scripts
  capture_hand_tracking_simple.py  # Original monolithic implementation
  capture_hand_tracking.py         # Extended original implementation
  hand_detection_helper.py         # Original helper functions
```

## Module Interfaces

### Camera Module
- **LibcameraCapture**: Provides frame capture from Raspberry Pi camera
  - `read()`: Returns (success, frame) tuple
  - `release()`: Cleans up camera resources

- **FrameProcessor**: Preprocesses frames for detection
  - `preprocess(frame)`: Applies cropping and scaling to frames

### Detection Module
- **ModelLoader**: Handles TensorFlow Lite model loading
  - `load_model()`: Loads model from file
  - `get_input_details()`: Returns input tensor specifications
  - `get_output_details()`: Returns output tensor specifications

- **InferenceEngine**: Runs model inference
  - `prepare_input(frame)`: Prepares frame data for model input
  - `run_inference(input_tensor)`: Executes model and returns raw results

- **LandmarkExtractor**: Processes model output
  - `extract_landmarks(...)`: Converts raw model output to landmark coordinates

### Tracking Module
- **HandTracker**: Tracks hand presence over time
  - `update(position, confidence)`: Updates tracking state
  - `is_tracking_stable()`: Returns whether tracking is stable
  - `get_tracking_quality()`: Returns quality metric (0-1)

- **PositionValidator**: Validates physical plausibility of detection
  - `is_valid(position)`: Returns whether position is physically plausible
  - `is_suspiciously_still(threshold)`: Detects false positive detections

- **CoordinateSmoother**: Smooths tracking coordinates
  - `exponential_smooth(position)`: Applies smoothing filter

### Visualization Module
- **Renderer**: Handles drawing landmarks and overlays
  - `draw_landmarks(frame, landmarks)`: Renders landmarks on frame
  - `draw_wrist(frame, wrist_position)`: Draws wrist position indicator

- **StatusDisplay**: Shows status information
  - `render_status(frame, params)`: Adds status overlay to frame

## Key Improvements Made

1. **Modular Architecture** - Restructured into clear responsibility modules
2. **Optimized LibcameraCapture** - Multi-threaded frame capture with efficient buffering
3. **Enhanced Detection Pipeline** - Separate model loading, inference, and landmark extraction
4. **Robust Tracking System** - Comprehensive hand tracking with position validation and smoothing
5. **Flexible Visualization Layer** - Configurable display options and rendering pipeline
6. **Clean API Design** - Clear interfaces between components for extensibility

## Extending the System

The modular architecture makes it easy to extend or modify the system:

### Adding a New Camera Implementation

1. Create a new class in the `camera/` directory that follows the same interface as `LibcameraCapture`
2. Implement the `read()` and `release()` methods
3. Update the `main.py` to use your new camera implementation

### Supporting a Different Model

1. Extend the `ModelLoader` class in `detection/model_loader.py` to handle your model format
2. Update the `InferenceEngine` class in `detection/inference_engine.py` for model-specific processing
3. Modify the `LandmarkExtractor` class as needed for the new landmark format

### Creating Custom Visualization

1. Create a new renderer class in the `visualization/` directory
2. Update the `FrameRenderer` class to use your custom visualization
3. Modify the `main.py` to initialize your new renderer

## Development Notes

### Code Cleanup

The project has been recently restructured from a monolithic script to a modular architecture:

1. **Original Structure**: A single large script (`capture_hand_tracking_simple.py`) with all functionality
2. **New Structure**: Modular design with clear separation of concerns
3. **Benefits**:
   - Improved maintainability
   - Better code reusability
   - Easier extensibility
   - Clear interfaces between components

### Original Implementation Notes

- Original scripts have been moved to an archive directory for reference
- The system uses `libcamera` as it's the modern standard for Raspberry Pi 5
- Threading separates frame capture from processing for better performance
- Resource cleanup ensures proper process termination and temp file removal

## License

This project is provided as-is for educational and development purposes.
