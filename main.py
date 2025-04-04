# main.py
"""
Main script to run the multi-clip, reframing video visualizer processing.
"""
import os
import sys

# Import project modules
import config
import utils
import audio_analyzer
import video_processor
import cropping # Need to check cascade path here too

def run():
    """Executes the main workflow."""
    print("--- Starting Multi-Clip Reframing Video Visualizer Processing ---")

    # 1. Prerequisite Checks
    if not utils.is_ffmpeg_available():
        print("Error: ffmpeg command not found.", file=sys.stderr); return False

    if not isinstance(config.INPUT_VIDEO_PATHS, list) or not config.INPUT_VIDEO_PATHS:
         print("Error: config.INPUT_VIDEO_PATHS must be a non-empty list.", file=sys.stderr); return False

    found_valid_video = False
    for path in config.INPUT_VIDEO_PATHS:
         if os.path.exists(path): found_valid_video = True; break
    if not found_valid_video:
         print(f"Error: None of the video files in config.INPUT_VIDEO_PATHS found.", file=sys.stderr); return False

    if not os.path.exists(config.INPUT_AUDIO_PATH):
         print(f"Error: Input audio file not found: '{config.INPUT_AUDIO_PATH}'", file=sys.stderr); return False

    # Check for Haar Cascade if dynamic focus is enabled
    if config.ENABLE_DYNAMIC_FOCUS and not os.path.exists(config.HAAR_CASCADE_PATH):
        print(f"Error: Haar Cascade file not found at '{config.HAAR_CASCADE_PATH}'.", file=sys.stderr)
        print("Dynamic focus requires this file. Download it or disable dynamic focus in config.py.", file=sys.stderr)
        return False
    # Check if target aspect ratio is parsable (optional early check)
    if config.TARGET_ASPECT_RATIO and cropping.parse_aspect_ratio(config.TARGET_ASPECT_RATIO) is None:
         print(f"Warning: Invalid TARGET_ASPECT_RATIO format: '{config.TARGET_ASPECT_RATIO}'. Should be 'W:H'. Cropping might be disabled.", file=sys.stderr)


    print("Prerequisites met.")

    # 2. Analyze Audio
    try:
        times, intensity = audio_analyzer.calculate_onset_strength_curve(config.INPUT_AUDIO_PATH)
        if times is None or intensity is None or len(times) == 0:
             print("Error: Audio analysis did not produce valid results.", file=sys.stderr); return False
    except Exception as e:
        print(f"Error during audio analysis: {e}", file=sys.stderr); return False

    # 3. Process Video
    print("Starting video processing stage...")
    success = video_processor.process_video(times, intensity)

    # 4. Final Message
    if success:
        print(f"\n--- Processing Complete ---")
        print(f"Output video saved to: {config.OUTPUT_VIDEO_PATH}")
        return True
    else:
        print(f"\n--- Processing Failed ---", file=sys.stderr)
        print("Please check the error messages above.")
        return False


if __name__ == "__main__":
    if run():
        sys.exit(0) # Success
    else:
        sys.exit(1) # Failure