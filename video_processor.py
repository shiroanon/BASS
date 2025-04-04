# video_processor.py
"""
Handles reading/writing video, applying effects/transitions, cropping, and focus panning.
Uses FFmpeg for final muxing.
"""
import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import os
import tempfile
import time # For focus detection interval

# Import project modules
import config
import effects
import transitions
import utils
import cropping # New import
import focus_detector # New import

# (Keep _mux_with_ffmpeg as is)
def _mux_with_ffmpeg(temp_video_path, audio_path, output_path):
    """Combines video and audio using FFmpeg."""
    print(f"Combining video ('{temp_video_path}') and audio ('{audio_path}') using ffmpeg...")
    ffmpeg_command = [ 'ffmpeg', '-y', '-loglevel', config.FFMPEG_LOG_LEVEL, '-i', temp_video_path, '-i', audio_path, '-c:v', 'copy', '-c:a', 'aac', '-strict', 'experimental', '-map', '0:v:0', '-map', '1:a:0', '-shortest', '-pix_fmt', 'yuv420p', output_path]
    print(f"Executing FFmpeg command: {' '.join(ffmpeg_command)}")
    try:
        process = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')
        print("FFmpeg stderr:", process.stderr)
        print(f"Final video successfully created: {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg execution (return code {e.returncode}):")
        print("FFmpeg stdout:", e.stdout); print("FFmpeg stderr:", e.stderr)
        return False
    except FileNotFoundError: print("Error: ffmpeg command not found."); return False


def _get_first_clip_properties(video_paths):
    """Gets FPS, Width, Height from the first valid video clip."""
    # (This function remains the same)
    for path in video_paths:
        if not os.path.exists(path): continue
        cap = cv2.VideoCapture(path)
        if not cap.isOpened(): continue
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if fps > 0 and width > 0 and height > 0:
            print(f"Using properties from first valid clip ({path}): {width}x{height} @ {fps:.2f} FPS")
            return fps, width, height
        else: print(f"Warning: Invalid properties in {path}. Skipping.")
    print("Error: Could not determine valid video properties.")
    return None, None, None

def process_video(audio_times, onset_strength_curve):
    """
    Reads multiple input videos, applies effects, cropping, focus panning, transitions,
    writes temp video, then muxes with audio.
    """
    # --- Initialize Focus Detector ---
    detector = None
    if config.ENABLE_DYNAMIC_FOCUS:
        try:
            detector = focus_detector.FocusDetector()
        except Exception as e:
            print(f"Error initializing Focus Detector: {e}")
            print("Disabling dynamic focus.")
            config.ENABLE_DYNAMIC_FOCUS = False # Disable if init fails

    # --- Get Base Video Properties ---
    base_fps, base_orig_w, base_orig_h = _get_first_clip_properties(config.INPUT_VIDEO_PATHS)
    if base_fps is None: return False

    # --- Calculate Crop / Output Dimensions ---
    crop_w, crop_h, output_w, output_h = cropping.calculate_crop_dimensions(
        base_orig_w, base_orig_h, config.TARGET_ASPECT_RATIO
    )
    aspect_change_enabled = (output_w != base_orig_w or output_h != base_orig_h)
    if not aspect_change_enabled:
        print("Note: Output dimensions match original. Cropping/panning disabled.")
        config.ENABLE_DYNAMIC_FOCUS = False # Disable focus if not cropping

    # --- Setup Video Output (Temporary) ---
    temp_dir = tempfile.gettempdir()
    temp_video_path = os.path.join(temp_dir, config.TEMP_VIDEO_FILENAME)
    print(f"Using temporary video file: {temp_video_path}")
    fourcc = cv2.VideoWriter_fourcc(*config.VIDEO_CODEC)
    # *** IMPORTANT: Initialize writer with the TARGET output dimensions ***
    video_writer = cv2.VideoWriter(temp_video_path, fourcc, base_fps, (output_w, output_h))
    if not video_writer.isOpened():
         print(f"Error: Could not open VideoWriter for temporary file: {temp_video_path}")
         return False

    # --- Process Clips ---
    global_frame_index = 0
    last_processed_frame_for_transition = None # Store the final *cropped* frame
    total_processed_frames = 0
    processing_success = True

    # -- Focus Panning State --
    # Coordinates relative to the *original* frame dimensions (base_orig_w, base_orig_h)
    current_focus_center_x = base_orig_w / 2.0
    current_focus_center_y = base_orig_h / 2.0
    target_focus_center_x = base_orig_w / 2.0
    target_focus_center_y = base_orig_h / 2.0
    last_focus_detect_time = -config.FOCUS_DETECT_INTERVAL_SEC # Ensure detection on first frame

    try:
        num_clips = len(config.INPUT_VIDEO_PATHS)
        for clip_idx, clip_path in enumerate(config.INPUT_VIDEO_PATHS):
            print(f"\n--- Processing Clip {clip_idx + 1}/{num_clips}: {clip_path} ---")
            if not os.path.exists(clip_path): continue # Skip missing
            cap = cv2.VideoCapture(clip_path)
            if not cap.isOpened(): continue # Skip unopenable

            clip_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Note: We ignore clip's specific FPS/dimensions here, assume we force to base_fps/base_orig_w/h before cropping

            # --- Handle Transition ---
            transition_applied = False
            if clip_idx > 0 and config.TRANSITION_TYPE != 'none' and last_processed_frame_for_transition is not None:
                print(f"Generating '{config.TRANSITION_TYPE}' transition...")
                ret, first_frame_raw = cap.read() # Read first frame of *new* clip
                if ret:
                    # Process this first frame (crop, effects) to get the transition target
                    # Resize to base dimensions first if necessary
                    if first_frame_raw.shape[1] != base_orig_w or first_frame_raw.shape[0] != base_orig_h:
                        first_frame_raw = cv2.resize(first_frame_raw, (base_orig_w, base_orig_h), interpolation=cv2.INTER_LINEAR)

                    # Crop the first frame using *current* pan state
                    first_frame_cropped = cropping.crop_frame(
                        first_frame_raw, current_focus_center_x, current_focus_center_y,
                        crop_w, crop_h, output_w, output_h
                    )
                    # Apply effects to the cropped first frame
                    current_global_time = global_frame_index / base_fps # Time *before* transition starts
                    onset_idx = min(np.searchsorted(audio_times, current_global_time, side="left"), len(onset_strength_curve) - 1)
                    first_frame_processed = effects.apply_configured_effects(
                        first_frame_cropped, onset_strength_curve[onset_idx], output_w, output_h
                    )

                    # Generate transition frames
                    transition_frames_count = max(1, int(config.TRANSITION_DURATION_SEC * base_fps))
                    for i in range(transition_frames_count):
                        progress = (i + 1) / transition_frames_count
                        # Transition between the *final processed* frames
                        t_frame = transitions.get_transition_frame(
                            last_processed_frame_for_transition, # Already processed/cropped
                            first_frame_processed,            # Also processed/cropped
                            progress, config.TRANSITION_TYPE
                        )
                        video_writer.write(t_frame)
                        global_frame_index += 1
                        total_processed_frames += 1
                    transition_applied = True
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 1) # Skip frame 0 as it was used
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Start from frame 0 if transition failed


            # --- Process Frames within the Current Clip ---
            start_frame_in_clip = 1 if transition_applied else 0
            for frame_in_clip_idx in tqdm(range(start_frame_in_clip, clip_total_frames), desc=f"Clip {clip_idx + 1} Frames"):
                ret, frame_raw = cap.read()
                if not ret: break

                # A. Resize raw frame to base dimensions if needed
                if frame_raw.shape[1] != base_orig_w or frame_raw.shape[0] != base_orig_h:
                    frame_resized = cv2.resize(frame_raw, (base_orig_w, base_orig_h), interpolation=cv2.INTER_LINEAR)
                else:
                    frame_resized = frame_raw # No resize needed

                current_global_time = global_frame_index / base_fps

                # B. Dynamic Focus Detection & Panning (if enabled)
                if config.ENABLE_DYNAMIC_FOCUS and detector:
                    # Time to check focus?
                    if current_global_time - last_focus_detect_time >= config.FOCUS_DETECT_INTERVAL_SEC:
                        # print(f"Time {current_global_time:.2f}: Detecting focus...")
                        focus_point = detector.detect_focus_center(frame_resized) # Detect on resized frame
                        if focus_point:
                            target_focus_center_x, target_focus_center_y = focus_point
                            # print(f"  New target focus: ({target_focus_center_x:.0f}, {target_focus_center_y:.0f})")
                        else:
                            # No focus detected, gradually return to center? Or stay? Let's return to center.
                            target_focus_center_x = base_orig_w / 2.0
                            target_focus_center_y = base_orig_h / 2.0
                            # print("  No focus detected, targeting center.")
                        last_focus_detect_time = current_global_time

                    # Smoothly update current focus center towards target
                    delta_x = target_focus_center_x - current_focus_center_x
                    delta_y = target_focus_center_y - current_focus_center_y
                    current_focus_center_x += delta_x * config.PAN_SMOOTHING_FACTOR
                    current_focus_center_y += delta_y * config.PAN_SMOOTHING_FACTOR
                    # print(f"  Current focus: ({current_focus_center_x:.0f}, {current_focus_center_y:.0f})")


                # C. Crop the frame (using current focus center)
                frame_cropped = cropping.crop_frame(
                    frame_resized, current_focus_center_x, current_focus_center_y,
                    crop_w, crop_h, output_w, output_h
                )

                # D. Apply visual effects (Zoom, BCS) to the cropped frame
                onset_idx = min(np.searchsorted(audio_times, current_global_time, side="left"), len(onset_strength_curve) - 1)
                frame_final_processed = effects.apply_configured_effects(
                    frame_cropped, onset_strength_curve[onset_idx], output_w, output_h # Use output dims here
                )

                # E. Write final frame
                video_writer.write(frame_final_processed)
                global_frame_index += 1
                total_processed_frames += 1

                # Store for next potential transition
                last_processed_frame_for_transition = frame_final_processed

            cap.release()

    except Exception as e:
        print(f"\nError during video processing loop: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        processing_success = False
    finally:
        video_writer.release()
        print(f"\nTemporary video processing finished. Total frames written: {total_processed_frames}.")

    if total_processed_frames == 0 and processing_success:
        print("Error: No frames were processed or written.")
        processing_success = False

    if not processing_success:
        utils.cleanup_file(temp_video_path)
        return False

    # --- Mux Audio and Video ---
    mux_success = _mux_with_ffmpeg(temp_video_path, config.INPUT_AUDIO_PATH, config.OUTPUT_VIDEO_PATH)

    # --- Cleanup ---
    if mux_success:
        utils.cleanup_file(temp_video_path)
        return True
    else:
        print(f"Muxing failed. Temporary video file kept: {temp_video_path}")
        return False