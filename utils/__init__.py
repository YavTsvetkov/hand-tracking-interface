"""
Utility modules for the hand tracking application.
"""

from .arg_parser import parse_args
from .timing_utils import FPSCounter, Timer, PerformanceTracker
from .math_utils import (
    calculate_distance,
    calculate_speed,
    calculate_angle,
    normalize_vector,
    calculate_centroid,
    clip_value
)
from .debug_logger import DebugLogger

__all__ = [
    'parse_args',
    'FPSCounter', 
    'Timer', 
    'PerformanceTracker',
    'calculate_distance',
    'calculate_speed',
    'calculate_angle',
    'normalize_vector',
    'calculate_centroid',
    'clip_value',
    'DebugLogger'
]