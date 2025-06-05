"""
Timing and performance measurement utilities.
"""

import time
from collections import deque


class FPSCounter:
    """FPS measurement with rolling average."""
    
    def __init__(self, window_size=30):
        """Initialize FPS counter with specified window size."""
        self.prev_frame_time = None
        self.frame_times = deque(maxlen=window_size)
    
    def update(self):
        """Update frame timing information."""
        current_time = time.time()
        
        if self.prev_frame_time is not None:
            # Calculate frame time
            frame_time = current_time - self.prev_frame_time
            self.frame_times.append(frame_time)
            
        self.prev_frame_time = current_time
    
    def get_fps(self):
        """Get current FPS based on rolling window."""
        if not self.frame_times:
            return 0.0
            
        avg_frame_time = sum(self.frame_times) / len(self.frame_times)
        
        if avg_frame_time > 0:
            return 1.0 / avg_frame_time
        else:
            return 0.0
            
    def reset(self):
        """Reset FPS counter."""
        self.frame_times.clear()
        self.prev_frame_time = None


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        """Initialize timer."""
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop the timer."""
        self.end_time = time.time()
        return self
    
    def elapsed(self):
        """Get elapsed time in seconds."""
        if self.start_time is None:
            return 0.0
            
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def elapsed_ms(self):
        """Get elapsed time in milliseconds."""
        return self.elapsed() * 1000.0


class PerformanceTracker:
    """Track performance metrics for different parts of the application."""
    
    def __init__(self):
        """Initialize performance tracker."""
        self.timers = {}
        self.stats = {}
    
    def start_timer(self, name):
        """Start a named timer."""
        timer = Timer()
        self.timers[name] = timer
        timer.start()
        return timer
    
    def stop_timer(self, name):
        """Stop a named timer and update statistics."""
        if name not in self.timers:
            return 0.0
            
        timer = self.timers[name]
        timer.stop()
        elapsed = timer.elapsed_ms()
        
        # Update statistics
        if name not in self.stats:
            self.stats[name] = {'count': 0, 'total': 0.0, 'min': elapsed, 'max': elapsed}
            
        stats = self.stats[name]
        stats['count'] += 1
        stats['total'] += elapsed
        stats['min'] = min(stats['min'], elapsed)
        stats['max'] = max(stats['max'], elapsed)
        
        return elapsed
    
    def get_stats(self):
        """Get performance statistics."""
        result = {}
        
        for name, stats in self.stats.items():
            if stats['count'] > 0:
                avg = stats['total'] / stats['count']
                result[name] = {
                    'count': stats['count'],
                    'avg_ms': avg,
                    'min_ms': stats['min'],
                    'max_ms': stats['max']
                }
                
        return result
    
    def reset(self):
        """Reset all performance statistics."""
        self.timers.clear()
        self.stats.clear()