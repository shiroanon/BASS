# effects.py
"""
Functions to apply visual effects to video frames.
"""
import cv2
import numpy as np
import config # Import config settings

def _apply_zoom(frame, scale_factor, frame_width, frame_height):
    """Applies center zoom (scaling + cropping/padding). Internal helper."""
    if abs(scale_factor - 1.0) < 1e-6:
        return frame

    scaled_h = int(frame_height * scale_factor)
    scaled_w = int(frame_width * scale_factor)

    try:
        scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

        if scale_factor > 1.0: # Zoom In
            start_x = max(0, (scaled_w - frame_width) // 2)
            start_y = max(0, (scaled_h - frame_height) // 2)
            crop_h = min(frame_height, scaled_h - start_y)
            crop_w = min(frame_width, scaled_w - start_x)
            zoomed_frame = scaled_frame[start_y : start_y + crop_h, start_x : start_x + crop_w]
            if zoomed_frame.shape[0] != frame_height or zoomed_frame.shape[1] != frame_width:
                return cv2.resize(zoomed_frame, (frame_width, frame_height), interpolation=cv2.INTER_AREA)
            else:
                return zoomed_frame
        else: # Zoom Out
            start_x = (frame_width - scaled_w) // 2
            start_y = (frame_height - scaled_h) // 2
            padded_frame = np.zeros((frame_height, frame_width, 3), dtype=frame.dtype)
            padded_frame[start_y:start_y + scaled_h, start_x:start_x + scaled_w] = scaled_frame
            return padded_frame
    except cv2.error as e:
        print(f"Warning: OpenCV error during zoom (scale={scale_factor:.2f}): {e}. Skipping zoom.")
        return frame # Return original frame on error


def _apply_brightness_contrast(frame, brightness_shift, contrast_factor):
    """Applies brightness and contrast adjustment. Internal helper."""
    if abs(brightness_shift) < 1e-6 and abs(contrast_factor - 1.0) < 1e-6:
        return frame
    try:
        frame_float = frame.astype(np.float32)
        adjusted_frame_float = frame_float * contrast_factor + brightness_shift
        return np.clip(adjusted_frame_float, 0, 255).astype(np.uint8)
    except cv2.error as e:
        print(f"Warning: OpenCV error during brightness/contrast: {e}. Skipping.")
        return frame

def _apply_saturation(frame, saturation_factor):
    """Applies saturation adjustment. Internal helper."""
    if abs(saturation_factor - 1.0) < 1e-6:
        return frame
    try:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_frame)
        s_float = s.astype(np.float32) * saturation_factor
        s_adjusted = np.clip(s_float, 0, 255).astype(np.uint8)
        hsv_adjusted = cv2.merge([h, s_adjusted, v])
        return cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2BGR)
    except cv2.error as e:
        print(f"Warning: OpenCV error during saturation: {e}. Skipping.")
        return frame


def apply_configured_effects(frame, onset_intensity, frame_width, frame_height):
    """
    Applies effects configured in config.py based on onset intensity.

    Args:
        frame: The input video frame (NumPy array BGR).
        onset_intensity: Normalized onset strength (0-1) for this frame's time.
        frame_width: Width of the frame.
        frame_height: Height of the frame.

    Returns:
        The processed video frame (NumPy array BGR).
    """
    processed_frame = frame.copy() # Start with a copy

    # --- Apply Zoom ---
    if config.ENABLE_ZOOM:
        target_scale = config.BASE_SCALE + onset_intensity * config.ZOOM_SENSITIVITY
        zoom_scale = np.clip(target_scale, config.MIN_SCALE, config.MAX_SCALE)
        processed_frame = _apply_zoom(processed_frame, zoom_scale, frame_width, frame_height)

    # --- Apply Brightness/Contrast ---
    # Note: Applying B/C before saturation is usually preferred
    brightness_shift = 0.0
    contrast_factor = 1.0
    if config.ENABLE_BRIGHTNESS:
        target_brightness = config.MIN_BRIGHTNESS + onset_intensity * config.BRIGHTNESS_SENSITIVITY
        brightness_shift = np.clip(target_brightness, config.MIN_BRIGHTNESS, config.MAX_BRIGHTNESS)

    if config.ENABLE_CONTRAST:
        target_contrast = config.MIN_CONTRAST + onset_intensity * config.CONTRAST_SENSITIVITY
        contrast_factor = np.clip(target_contrast, config.MIN_CONTRAST, config.MAX_CONTRAST)

    if config.ENABLE_BRIGHTNESS or config.ENABLE_CONTRAST:
         processed_frame = _apply_brightness_contrast(processed_frame, brightness_shift, contrast_factor)

    # --- Apply Saturation ---
    if config.ENABLE_SATURATION:
        target_saturation = config.MIN_SATURATION + onset_intensity * config.SATURATION_SENSITIVITY
        saturation_factor = np.clip(target_saturation, config.MIN_SATURATION, config.MAX_SATURATION)
        processed_frame = _apply_saturation(processed_frame, saturation_factor)

    return processed_frame