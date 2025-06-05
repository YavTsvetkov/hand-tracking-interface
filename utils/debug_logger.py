"""
Debugging logging utilities.
"""

import logging
import os
import sys
import time


class DebugLogger:
    """Debug logging with different verbosity levels."""
    
    # Log levels
    LEVEL_ERROR = 0
    LEVEL_WARNING = 1
    LEVEL_INFO = 2
    LEVEL_DEBUG = 3
    LEVEL_TRACE = 4
    
    def __init__(self, level=LEVEL_INFO, log_to_file=False, filename="hand_tracking_debug.log"):
        """Initialize debug logger with specified verbosity level."""
        self.level = level
        self.log_to_file = log_to_file
        self.filename = filename
        
        # Configure logging
        log_format = '%(asctime)s [%(levelname)s] %(message)s'
        date_format = '%Y-%m-%d %H:%M:%S'
        
        # Set up logging
        logging.basicConfig(
            level=logging.DEBUG,
            format=log_format,
            datefmt=date_format
        )
        
        # Create logger
        self.logger = logging.getLogger('hand_tracking')
        
        # Add file handler if requested
        if log_to_file:
            file_handler = logging.FileHandler(filename)
            file_handler.setFormatter(logging.Formatter(log_format))
            self.logger.addHandler(file_handler)
            
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)
    
    def error(self, message):
        """Log error message."""
        if self.level >= self.LEVEL_ERROR:
            self.logger.error(message)
    
    def warning(self, message):
        """Log warning message."""
        if self.level >= self.LEVEL_WARNING:
            self.logger.warning(message)
    
    def info(self, message):
        """Log info message."""
        if self.level >= self.LEVEL_INFO:
            self.logger.info(message)
    
    def debug(self, message):
        """Log debug message."""
        if self.level >= self.LEVEL_DEBUG:
            self.logger.debug(message)
    
    def trace(self, message):
        """Log trace message (very verbose)."""
        if self.level >= self.LEVEL_TRACE:
            self.logger.debug(f"[TRACE] {message}")
    
    def set_level(self, level):
        """Set the logging level."""
        self.level = level
        
    def log_performance(self, component, duration_ms):
        """Log performance information."""
        if self.level >= self.LEVEL_DEBUG:
            self.logger.debug(f"[PERF] {component}: {duration_ms:.2f}ms")
