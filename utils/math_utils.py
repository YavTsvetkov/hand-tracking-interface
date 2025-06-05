"""
Mathematical utilities for hand tracking.
"""

import math
import numpy as np


def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    if point1 is None or point2 is None:
        return float('inf')
    
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_speed(point1, point2, time_delta):
    """Calculate speed in pixels per second."""
    if point1 is None or point2 is None or time_delta == 0:
        return 0.0
    
    distance = calculate_distance(point1, point2)
    return distance / time_delta


def calculate_angle(point1, point2):
    """Calculate angle in degrees between two points (0-360)."""
    if point1 is None or point2 is None:
        return 0.0
    
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    
    angle = math.degrees(math.atan2(dy, dx)) % 360
    return angle


def normalize_vector(vector):
    """Normalize a vector to unit length."""
    if vector is None:
        return None
        
    magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
    
    if magnitude < 0.000001:  # Avoid division by zero
        return (0, 0)
        
    return (vector[0] / magnitude, vector[1] / magnitude)


def calculate_centroid(points):
    """Calculate the centroid (average position) of multiple points."""
    if not points:
        return None
    
    x_sum = sum(p[0] for p in points)
    y_sum = sum(p[1] for p in points)
    
    return (int(x_sum / len(points)), int(y_sum / len(points)))


def clip_value(value, min_val, max_val):
    """Clip a value to the specified range."""
    return max(min(value, max_val), min_val)