# audio_analyzer.py
"""
Handles audio loading and analysis (onset detection).
"""
import librosa
import numpy as np
import config # Import config settings

def calculate_onset_strength_curve(audio_path):
    """Loads audio and calculates a smoothed, normalized onset strength curve."""
    print(f"Loading audio: {audio_path}")
    try:
        # Use config settings for sample rate if needed, or let librosa handle it
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        raise

    print(f"Audio loaded: Sample Rate={sr} Hz, Duration={librosa.get_duration(y=y, sr=sr):.2f}s")

    print("Calculating onset strength envelope...")
    onset_env = librosa.onset.onset_strength(
        y=y,
        sr=sr,
        hop_length=config.HOP_LENGTH,
        n_fft=config.N_FFT,
        aggregate=config.ONSET_AGGREGATE
    )

    # Normalize
    max_onset = np.max(onset_env)
    if max_onset > 1e-6:
        normalized_onset = onset_env / max_onset
    else:
        normalized_onset = np.zeros_like(onset_env)

    # Smooth
    smoothing_window_frames = int(config.ONSET_SMOOTHING_SEC * sr / config.HOP_LENGTH)
    if smoothing_window_frames > 1:
        padded_onset = np.pad(normalized_onset, (smoothing_window_frames // 2, smoothing_window_frames // 2), mode='edge')
        smoothed_onset = np.convolve(padded_onset, np.ones(smoothing_window_frames)/smoothing_window_frames, mode='valid')
        # Adjust length if needed after convolution
        if len(smoothed_onset) < len(normalized_onset):
             pad_width = len(normalized_onset) - len(smoothed_onset)
             smoothed_onset = np.pad(smoothed_onset, (0, pad_width), mode='edge')
        elif len(smoothed_onset) > len(normalized_onset):
             smoothed_onset = smoothed_onset[:len(normalized_onset)]
    else:
        smoothed_onset = normalized_onset

    times = librosa.times_like(smoothed_onset, sr=sr, hop_length=config.HOP_LENGTH, n_fft=config.N_FFT)

    print(f"Onset strength curve calculated. Found {len(times)} values.")
    return times, smoothed_onset