# transitions.py
"""
Functions to generate transition frames between two video frames.
"""
import cv2
import numpy as np

def fade(frame1, frame2, progress):
    """
    Generates a cross-fade frame.

    Args:
        frame1: The outgoing frame (NumPy array BGR).
        frame2: The incoming frame (NumPy array BGR).
        progress: Float from 0.0 (frame1) to 1.0 (frame2).

    Returns:
        The blended frame (NumPy array BGR).
    """
    if frame1 is None or frame2 is None:
        return frame2 if frame1 is None else frame1 # Return the valid frame

    # Ensure frames have same dimensions (resize frame2 to match frame1 if needed)
    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    alpha = progress       # Weight for frame2
    beta = 1.0 - progress  # Weight for frame1
    try:
        return cv2.addWeighted(frame2, alpha, frame1, beta, 0.0)
    except cv2.error as e:
        print(f"Warning: OpenCV error during fade: {e}. Returning frame2.")
        return frame2


def swipe(frame1, frame2, progress, direction='left'):
    """
    Generates a swipe transition frame.

    Args:
        frame1: Outgoing frame.
        frame2: Incoming frame.
        progress: Float from 0.0 (frame1) to 1.0 (frame2).
        direction: 'left', 'right', 'up', or 'down'.

    Returns:
        The transition frame.
    """
    if frame1 is None or frame2 is None:
        return frame2 if frame1 is None else frame1

    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    rows, cols = frame1.shape[:2]
    output_frame = frame1.copy() # Start with the outgoing frame

    try:
        if direction == 'left':
            split_col = int(cols * (1.0 - progress))
            if split_col < cols: # Avoid index error if split_col == cols
                output_frame[:, split_col:] = frame2[:, split_col:]
        elif direction == 'right':
            split_col = int(cols * progress)
            if split_col > 0: # Avoid index error if split_col == 0
                output_frame[:, :split_col] = frame2[:, :split_col]
        elif direction == 'up':
            split_row = int(rows * (1.0 - progress))
            if split_row < rows:
                output_frame[split_row:, :] = frame2[split_row:, :]
        elif direction == 'down':
            split_row = int(rows * progress)
            if split_row > 0:
                output_frame[:split_row, :] = frame2[:split_row, :]
        else: # Default to frame2 if direction is invalid
            return frame2

        return output_frame
    except (cv2.error, IndexError) as e:
         print(f"Warning: Error during swipe ({direction}, progress={progress:.2f}): {e}. Returning frame2.")
         return frame2

def get_transition_frame(frame1, frame2, progress, transition_type):
    """
    Calls the appropriate transition function based on type.
    """
    if transition_type == 'fade':
        return fade(frame1, frame2, progress)
    elif transition_type == 'swipe_left':
        return swipe(frame1, frame2, progress, direction='left')
    elif transition_type == 'swipe_right':
        return swipe(frame1, frame2, progress, direction='right')
    elif transition_type == 'swipe_up':
        return swipe(frame1, frame2, progress, direction='up')
    elif transition_type == 'swipe_down':
        return swipe(frame1, frame2, progress, direction='down')
    else: # 'none' or invalid type
        # Return incoming frame immediately after progress > 0 ? or just frame1? Let's return frame1 until progress > 0.5
        return frame2 if progress >= 0.5 else frame1