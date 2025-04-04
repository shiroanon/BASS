# cropping.py
"""
Functions for aspect ratio calculation and frame cropping.
"""
import cv2
import numpy as np

def parse_aspect_ratio(ratio_str):
    """Parses 'W:H' string to a float ratio (Width / Height)."""
    if not ratio_str or ':' not in ratio_str:
        return None
    try:
        w, h = map(float, ratio_str.split(':'))
        if h == 0: return None
        return w / h
    except ValueError:
        return None

def calculate_crop_dimensions(orig_w, orig_h, target_aspect_ratio_str):
    """
    Calculates the dimensions needed to crop the original frame to the target ratio.

    Returns:
        (crop_w, crop_h, target_w, target_h):
          crop_w, crop_h: Dimensions of the rectangle to extract from original.
          target_w, target_h: Final dimensions of the output video frame.
    """
    target_ratio = parse_aspect_ratio(target_aspect_ratio_str)
    if target_ratio is None: # No target or invalid format, use original
        print("No valid target aspect ratio. Using original dimensions.")
        return orig_w, orig_h, orig_w, orig_h

    orig_ratio = orig_w / orig_h

    if abs(target_ratio - orig_ratio) < 1e-6: # Target is same as original
        return orig_w, orig_h, orig_w, orig_h

    if target_ratio > orig_ratio:
        # Target is wider than original (relative), crop height (pillarbox)
        crop_h = int(orig_w / target_ratio)
        crop_w = orig_w
        target_w = orig_w # Output width matches original width
        target_h = crop_h # Output height is the cropped height
    else:
        # Target is taller than original (relative), crop width (letterbox)
        crop_w = int(orig_h * target_ratio)
        crop_h = orig_h
        target_w = crop_w # Output width is the cropped width
        target_h = orig_h # Output height matches original height

    print(f"Target Aspect Ratio: {target_aspect_ratio_str} ({target_ratio:.3f})")
    print(f"Original Dimensions: {orig_w}x{orig_h} (Ratio: {orig_ratio:.3f})")
    print(f"Crop Dimensions (from original): {crop_w}x{crop_h}")
    print(f"Output Frame Dimensions: {target_w}x{target_h}")

    return crop_w, crop_h, target_w, target_h


def crop_frame(frame, crop_center_x, crop_center_y, crop_w, crop_h, target_w, target_h):
    """
    Crops the frame based on the desired center and dimensions.

    Args:
        frame: The original input frame (NumPy array BGR).
        crop_center_x: Desired center X of the crop (relative to original frame).
        crop_center_y: Desired center Y of the crop (relative to original frame).
        crop_w: Width of the rectangle to crop from the original frame.
        crop_h: Height of the rectangle to crop from the original frame.
        target_w: Final output width (should match crop_w if cropping width).
        target_h: Final output height (should match crop_h if cropping height).

    Returns:
        The cropped (and potentially resized if needed) frame (NumPy array BGR).
    """
    orig_h, orig_w = frame.shape[:2]

    # Calculate top-left corner of the crop box
    crop_x = int(round(crop_center_x - crop_w / 2.0))
    crop_y = int(round(crop_center_y - crop_h / 2.0))

    # --- Boundary Checks ---
    # Ensure crop coordinates are within the original frame bounds
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)

    # Ensure the crop doesn't go *beyond* the frame dimensions
    if crop_x + crop_w > orig_w:
        crop_x = orig_w - crop_w
    if crop_y + crop_h > orig_h:
        crop_y = orig_h - crop_h

    # Second check after potential adjustment, clamp again if crop dim > orig dim
    crop_x = max(0, crop_x)
    crop_y = max(0, crop_y)
    actual_crop_w = min(crop_w, orig_w - crop_x)
    actual_crop_h = min(crop_h, orig_h - crop_y)
    # --- End Boundary Checks ---

    # Extract the crop
    try:
        cropped = frame[crop_y : crop_y + actual_crop_h, crop_x : crop_x + actual_crop_w]

        # Resize if the actual crop size doesn't match the target output size
        # (This can happen if the crop window hit the edge)
        if cropped.shape[1] != target_w or cropped.shape[0] != target_h:
            # Use INTER_AREA for potentially downscaling, INTER_LINEAR for upscaling
            interpolation = cv2.INTER_AREA if (target_w * target_h < cropped.shape[1] * cropped.shape[0]) else cv2.INTER_LINEAR
            cropped = cv2.resize(cropped, (target_w, target_h), interpolation=interpolation)

        return cropped

    except Exception as e:
        print(f"Warning: Error during cropping: {e}. Returning black frame.")
        # Return a black frame of the target size on error
        return np.zeros((target_h, target_w, 3), dtype=frame.dtype)