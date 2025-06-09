# Hand Tracking System

A real-time hand tracking system using MediaPipe for robotics control applications. The system provides 4-quadrant Cartesian coordinate mapping with direct cmd_vel output suitable for robot navigation.

## Overview

This hand tracking system detects hand positions using MediaPipe and converts them to robotics control commands. The system features:

- MediaPipe Hands with 21-point landmark detection
- 4-quadrant Cartesian coordinate system
- Dead zone for stable center positioning
- Real-time cmd_vel output for ROS integration
- Visual overlay with coordinate grid and status information
- Exponential smoothing for position stability

## Installation

### Dependencies

Install the required Python packages:

```bash
pip install opencv-python mediapipe numpy
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### System Requirements

- Linux system (tested on Raspberry Pi 5)
- USB camera or Raspberry Pi camera module
- Python 3.8 or higher

## Usage

### Primary Application

Run the main hand tracking application:

```bash
python3 simple_hand_tracking.py
```

This application provides:
- Real-time hand detection and tracking
- 4-quadrant Cartesian coordinate system
- cmd_vel output for robotics control
- Visual interface with coordinate overlays

### Alternative Implementation

Run the alternative implementation with existing camera infrastructure:

```bash
python3 main.py --res 640x480
```

This implementation offers:
- Integration with modular camera system
- Same coordinate processing functionality
- ROS cmd_vel publishing capability

### Output Format

The system outputs robotics-compatible cmd_vel commands:

```python
{
    'linear_x': 0.25,      # Forward/backward velocity (-0.5 to 0.5 m/s)
    'angular_z': -0.15,    # Left/right turn velocity (-1.0 to 1.0 rad/s)
    'zone': 'forward',     # Current quadrant: forward/backward/left/right/center
    'intensity': 0.8       # Distance-based intensity scaling (0.0 to 1.0)
}
```

## Configuration

### Hand Tracker Settings

Configure the hand tracking parameters in the `HandTracker` class:

```python
HandTracker(
    camera_width=640,           # Camera resolution width
    camera_height=480,          # Camera resolution height
)

# MediaPipe Hands settings
min_detection_confidence=0.5,   # Minimum confidence for hand detection
min_tracking_confidence=0.5,    # Minimum confidence for hand tracking
max_num_hands=1,                # Maximum number of hands to track
```

### Cartesian Controller Settings

Configure the coordinate mapping in the `CartesianController` class:

```python
CartesianController(
    frame_width=640,            # Frame width for coordinate mapping
    frame_height=480,           # Frame height for coordinate mapping
    dead_zone=50,               # Dead zone radius in pixels
)

# Speed limits
max_linear_speed=0.5,           # Maximum forward/backward speed (m/s)
max_angular_speed=1.0,          # Maximum turning speed (rad/s)
```

### ROS Configuration

The system includes ROS-specific configuration in `config/ros_config.py`:

```python
# ROS node settings
DEFAULT_ROS_TOPIC = "/cmd_vel"
DEFAULT_ROS_NODE_NAME = "hand_tracking_controller"
DEFAULT_ROS_PUBLISH_RATE = 10.0

# Smoothing factors
CMD_SMOOTHING_FACTOR = 0.2      # For ROS cmd_vel output smoothing
HAND_SMOOTHING_FACTOR = 0.7     # For hand coordinate tracking
```

### Coordinate System

The system uses a standard Cartesian coordinate system:

- **Origin (0,0)**: Center of the frame
- **X-axis**: Horizontal (negative = left, positive = right)
- **Y-axis**: Vertical (negative = forward/up, positive = backward/down)
- **Dead Zone**: Circular area around origin where cmd_vel = 0
- **Quadrants**: Forward, Backward, Left, Right based on dominant axis

## System Architecture

### Core Components

The system consists of two primary classes:

1. **HandTracker**: 
   - Uses MediaPipe Hands for hand detection
   - Calculates hand center from 21 landmarks
   - Applies exponential smoothing for stability

2. **CartesianController**:
   - Maps hand coordinates to 4-quadrant system
   - Implements dead zone for stable stop commands
   - Generates cmd_vel output with distance-based scaling
## Troubleshooting

### Common Issues

1. **Camera not detected**:
   ```bash
   # Check camera connection
   v4l2-ctl --list-devices
   
   # Test with simple capture
   python3 -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.read()[0] else 'Camera failed')"
   ```

2. **MediaPipe installation issues**:
   ```bash
   # Install/reinstall MediaPipe
   pip uninstall mediapipe
   pip install mediapipe
   ```

3. **Hand not detected**:
   - Ensure good lighting conditions
   - Keep hand clearly visible in frame
   - Avoid complex backgrounds
   - Check MediaPipe confidence thresholds

4. **Coordinate jitter**:
   - Adjust smoothing_factor (0.7 = less smoothing, 0.3 = more smoothing)
   - Increase dead_zone radius for more stable center zone

### Performance Tips

- Good lighting improves detection accuracy
- Solid backgrounds work better than complex patterns
- Centered hand position provides optimal tracking
- Steady hand movement ensures stable coordinates

## Project Structure

```
hand_tracking/
├── simple_hand_tracking.py    # Main hand tracking application
├── main.py           # Alternative implementation
├── requirements.txt           # Python dependencies
├── README.md                  # Documentation
├── camera/                    # Camera handling modules
│   ├── __init__.py
│   ├── frame_processor.py     # Frame preprocessing utilities
│   └── libcamera_capture.py   # Camera capture interface
├── config/                    # Configuration management
│   ├── __init__.py
│   ├── ros_config.py          # ROS-specific settings
│   └── settings.py            # Application settings
├── ros_cmd_vel/               # ROS command velocity integration
│   ├── __init__.py
│   ├── coordinate_parser.py   # Coordinate processing
│   ├── integration.py         # ROS integration
│   ├── ros_publisher.py       # ROS publishing
│   └── ros_requirements.txt   # ROS-specific dependencies
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── arg_parser.py          # Command line argument parsing
│   ├── debug_logger.py        # Debug logging utilities
│   ├── math_utils.py          # Mathematical utilities
│   └── timing_utils.py        # Performance timing utilities
└── archive/                   # Archive directory
```

## ROS Integration

For ROS applications, the coordinate output can be mapped to geometry_msgs/Twist:

```python
from geometry_msgs.msg import Twist

def publish_cmd_vel(cmd_vel_data):
    twist = Twist()
    twist.linear.x = cmd_vel_data['linear_x']
    twist.angular.z = cmd_vel_data['angular_z']
    cmd_vel_pub.publish(twist)
```

The `ros_cmd_vel/` directory contains ROS integration modules for immediate use.
## Getting Started

1. Install dependencies: `pip install opencv-python mediapipe numpy`
2. Run the application: `python3 simple_hand_tracking.py`
3. Test coordinate mapping by moving your hand through the quadrants
4. Observe cmd_vel output values for integration with your robotics application

## License

This project is provided as-is for educational and development purposes.
