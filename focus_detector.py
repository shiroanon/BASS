# focus_detector.py
"""
Detects the center of attention in a frame (using face detection).
"""
import cv2
import numpy as np
import os
import config # Import config settings

class FocusDetector:
    def __init__(self):
        """Initializes the face detector."""
        if not os.path.exists(config.HAAR_CASCADE_PATH):
            raise FileNotFoundError(
                f"Haar Cascade file not found at: {config.HAAR_CASCADE_PATH}\n"
                "Download it from OpenCV's GitHub repo or ensure the path is correct in config.py"
            )
        self.face_cascade = cv2.CascadeClassifier(config.HAAR_CASCADE_PATH)
        if self.face_cascade.empty():
             raise IOError(f"Failed to load Haar Cascade from {config.HAAR_CASCADE_PATH}")
        print("Face detector initialized.")
        self.last_known_focus = None # Store the last successful focus point

    def detect_focus_center(self, frame):
        """
        Detects faces and returns the center of the most prominent one(s).

        Args:
            frame: Input frame (NumPy array BGR).

        Returns:
            (center_x, center_y) relative to the frame dimensions,
            or None if no focus point is detected.
        """
        frame_h, frame_w = frame.shape[:2]
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate min face size based on frame height
        min_face_h = int(frame_h * config.FACE_DETECTION_MIN_SIZE_RATIO)
        min_face_w = int(frame_w * config.FACE_DETECTION_MIN_SIZE_RATIO) # Keep aspect roughly square-ish


        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=config.FACE_DETECTION_SCALE_FACTOR,
            minNeighbors=config.FACE_DETECTION_MIN_NEIGHBORS,
            minSize=(min(min_face_w, min_face_h), min(min_face_w, min_face_h)) # Use the smaller dim for minSize tuple
             # flags=cv2.CASCADE_SCALE_IMAGE # Default flag
        )

        if len(faces) == 0:
            # No faces detected, maybe return last known focus? Or center?
            # Let's return None for now, processor can decide fallback.
             # print("No faces detected.")
            return None

        # If multiple faces, find the largest one or center of cluster?
        # Strategy: Find center of the bounding box containing all faces
        if len(faces) > 1:
            # Combine bounding boxes
            min_x = min(x for x, y, w, h in faces)
            min_y = min(y for x, y, w, h in faces)
            max_x_w = max(x + w for x, y, w, h in faces)
            max_y_h = max(y + h for x, y, w, h in faces)
            center_x = (min_x + max_x_w) / 2.0
            center_y = (min_y + max_y_h) / 2.0
            # print(f"Multiple faces detected, using cluster center: ({center_x:.0f}, {center_y:.0f})")
        else:
            # Single face
            x, y, w, h = faces[0]
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            # print(f"Single face detected at: ({center_x:.0f}, {center_y:.0f})")

        self.last_known_focus = (center_x, center_y)
        return center_x, center_y