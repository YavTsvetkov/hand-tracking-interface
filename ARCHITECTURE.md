# Hand Tracking System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      main.py                               │
│                                                            │
│            HandTrackingApp                                 │
│            - Initializes components                        │
│            - Coordinates processing flow                   │
│            - Manages application lifecycle                 │
└─────────────────────────┬──────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────┬──────────────────────────────────┐
│                         │                                  │
▼                         ▼                                  ▼
┌────────────────┐  ┌────────────────┐            ┌────────────────┐
│   camera/      │  │  detection/    │            │ tracking/      │
│ - Capture      │  │ - Model load   │            │ - Hand state   │
│ - Processing   │──┼►- Inference    │───────────►│ - Validation   │
└────────────────┘  │ - Landmarks    │            │ - Smoothing    │
                    └────────────────┘            └───────┬────────┘
                                                          │
┌────────────────┐                                        │
│    utils/      │                                        │
│ - Arguments    │                                        │
│ - Logging      │◄────────────────────────────────────────
│ - Timing       │                       │
└────────────────┘                       │
                                         ▼
                               ┌────────────────┐
                               │visualization/  │
                               │ - Rendering    │
                               │ - Display      │
                               └────────────────┘
```

## Data Flow

1. Camera module captures frames
2. Detection module runs ML inference
3. Tracking module validates and tracks hand position
4. Visualization module renders output
5. Utils provide support functions

## Component Responsibilities

### Main Application (`main.py`)
- Application lifecycle management
- Component initialization and coordination
- Event processing and main loop

### Camera Module
- Hardware interfacing with Raspberry Pi camera
- Frame capture and preprocessing
- Format conversion (YUV to BGR)

### Detection Module
- TensorFlow Lite model management
- Hand landmark inference
- Coordinate extraction and normalization

### Tracking Module
- Hand state tracking
- Position validation and filtering
- Movement analysis and smoothing

### Visualization Module
- Landmark rendering
- Status information display
- UI elements and overlays

### Utils Module
- Command line argument parsing
- Performance measurement
- Debugging utilities

## Design Principles

1. **Single Responsibility**: Each module has a clear, focused purpose
2. **Dependency Injection**: Components receive their dependencies
3. **Interface Segregation**: Clean interfaces between components
4. **Open/Closed**: Easy to extend without modifying existing code
