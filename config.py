# config.py
"""
Configuration settings for the video visualizer.
"""

import numpy as np

# --- File Paths ---
INPUT_VIDEO_PATHS = ['clips/output.mp4'] # List of input clips
INPUT_AUDIO_PATH = 'audio/input_audio.opus'
OUTPUT_VIDEO_PATH = 'output_visualizer_reframed.mp4'
TEMP_VIDEO_FILENAME = "temp_video_reframed_output.mp4"

# --- Aspect Ratio & Cropping ---
# Target aspect ratio as "W:H" (e.g., "9:16", "1:1", "16:9", "4:3")
# Set to None or original aspect ratio string (e.g., "16:9") to disable cropping.
TARGET_ASPECT_RATIO = "9:16"

# --- Focus Detection & Panning ---
ENABLE_DYNAMIC_FOCUS = True # Set to False to always use center crop
FOCUS_DETECT_INTERVAL_SEC = 0.2 # How often to re-detect faces (seconds)
PAN_SMOOTHING_FACTOR = 0.05 # Controls how quickly the view pans (0.0-1.0, lower is slower)
# Face Detection Parameters (tune these for your video content)
HAAR_CASCADE_PATH = 'haarcascade_frontalface_default.xml' # Path to the XML file
FACE_DETECTION_SCALE_FACTOR = 1.1 # Parameter for cv2.detectMultiScale
FACE_DETECTION_MIN_NEIGHBORS = 5  # Parameter for cv2.detectMultiScale
FACE_DETECTION_MIN_SIZE_RATIO = 0.05 # Minimum face size relative to frame height

# --- Transitions ---
TRANSITION_TYPE = 'swipe_left' # 'none', 'fade', 'swipe_left', etc.
TRANSITION_DURATION_SEC = 0.5

# --- Audio Analysis Parameters ---
N_FFT = 2048
HOP_LENGTH = 512
ONSET_AGGREGATE = np.median
ONSET_SMOOTHING_SEC = 0.04

# --- Effect Configuration (unchanged) ---
ENABLE_ZOOM = True
ZOOM_SENSITIVITY = 0.25
# ... (rest of effect parameters remain the same) ...
MIN_SCALE = 0.95
MAX_SCALE = 1.15
BASE_SCALE = 1.0
ENABLE_BRIGHTNESS = True
BRIGHTNESS_SENSITIVITY = 25
MIN_BRIGHTNESS = 0
MAX_BRIGHTNESS = 50
ENABLE_CONTRAST = True
CONTRAST_SENSITIVITY = 0.4
MIN_CONTRAST = 1.0
MAX_CONTRAST = 1.6
ENABLE_SATURATION = True
SATURATION_SENSITIVITY = 0.5
MIN_SATURATION = 1.0
MAX_SATURATION = 1.8

# --- Video Writing Configuration ---
VIDEO_CODEC = 'mp4v'
FFMPEG_LOG_LEVEL = 'warning'