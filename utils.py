# utils.py
"""
Utility functions for the visualizer project.
"""
import shutil
import os

def is_ffmpeg_available():
    """Checks if the ffmpeg command is available in the system PATH."""
    return shutil.which("ffmpeg") is not None

def cleanup_file(filepath):
    """Removes a file if it exists, printing a message."""
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
            print(f"Cleaned up temporary file: {filepath}")
        except OSError as e:
            print(f"Warning: Could not remove temporary file {filepath}: {e}")