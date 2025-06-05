"""
Command line argument parsing utilities.
"""

import argparse
from config.settings import Settings

def parse_args():
    """Parse command line arguments for the hand tracking application."""
    parser = argparse.ArgumentParser(description="Hand Tracking System for Raspberry Pi")
    
    # Camera settings
    parser.add_argument('--res', type=str, default=Settings.DEFAULT_RESOLUTION, 
                       help=f'Camera resolution WxH (e.g., 640x480, 320x240, default: {Settings.DEFAULT_RESOLUTION})')
    parser.add_argument('--fps', type=int, default=Settings.DEFAULT_FPS,
                       help=f'Camera framerate (default: {Settings.DEFAULT_FPS})')
    
    # Model settings
    parser.add_argument('--model', type=str, default=Settings.DEFAULT_MODEL_PATH, 
                       help=f'Path to TFLite hand landmark model (default: {Settings.DEFAULT_MODEL_PATH})')
    
    # Detection settings
    parser.add_argument('--confidence', type=float, default=Settings.DEFAULT_CONFIDENCE_THRESHOLD,
                       help=f'Minimum confidence threshold (0-1, default: {Settings.DEFAULT_CONFIDENCE_THRESHOLD})')
    parser.add_argument('--smoothing', type=float, default=Settings.DEFAULT_SMOOTHING_FACTOR,
                       help=f'Coordinate smoothing factor (0-1, default: {Settings.DEFAULT_SMOOTHING_FACTOR})')
    parser.add_argument('--crop_factor', type=float, default=Settings.DEFAULT_CROP_FACTOR,
                       help=f'Center crop factor for better accuracy (0-1, default: {Settings.DEFAULT_CROP_FACTOR})')
    parser.add_argument('--max_jump', type=int, default=Settings.DEFAULT_MAX_JUMP,
                       help=f'Maximum pixel jump to filter noise (default: {Settings.DEFAULT_MAX_JUMP})')
    
    # Tracking settings
    parser.add_argument('--detection_loss_frames', type=int, default=Settings.DEFAULT_DETECTION_LOSS_FRAMES,
                       help=f'Number of frames before declaring hand lost (default: {Settings.DEFAULT_DETECTION_LOSS_FRAMES})')
    parser.add_argument('--stability_threshold', type=float, default=Settings.DEFAULT_STABILITY_THRESHOLD,
                       help=f'Minimum tracking quality for stable detection (0-1, default: {Settings.DEFAULT_STABILITY_THRESHOLD})')
    parser.add_argument('--false_positive_threshold', type=int, default=Settings.DEFAULT_FALSE_POSITIVE_THRESHOLD,
                       help=f'Frames of suspicious detection before rejection (default: {Settings.DEFAULT_FALSE_POSITIVE_THRESHOLD})')
    
    # Performance settings
    parser.add_argument('--frame_skip', type=int, default=Settings.DEFAULT_FRAME_SKIP, 
                       help=f'Process every Nth frame (default: {Settings.DEFAULT_FRAME_SKIP} for max accuracy)')
    
    # Display settings
    parser.add_argument('--headless', action='store_true', 
                       help='Run without GUI display')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    
    return parser.parse_args()