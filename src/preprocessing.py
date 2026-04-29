import numpy as np
from scipy import signal

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Enhanced Butterworth bandpass filter with higher order for sharper cutoff.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    # Use 'sos' (second-order sections) for better stability at higher orders
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    y = signal.sosfiltfilt(sos, data)
    return y

def notch_filter(data, cutoff, fs, q=30):
    """
    Notch filter to remove powerline noise.
    """
    nyq = 0.5 * fs
    w0 = cutoff / nyq
    b, a = signal.iirnotch(w0, q)
    y = signal.filtfilt(b, a, data)
    return y

def smooth_signal(data, window_ms=20, fs=500):
    """
    Applies a moving average to smooth out high-frequency 'chatter'.
    """
    window_size = int((window_ms / 1000.0) * fs)
    if window_size < 3: window_size = 3
    return np.convolve(data, np.ones(window_size)/window_size, mode='same')

def remove_baseline_wander(data, fs):
    """
    Robust baseline removal using a high-pass filter.
    """
    # 0.5 Hz is standard for removing slow-moving baseline wander
    return bandpass_filter(data, 0.5, 40, fs, order=3)

def preprocess_ecg(raw_signal, fs):
    """
    Advanced clinical preprocessing pipeline for extremely noisy signals.
    """
    # 1. Notch filter for 50Hz and 60Hz (Powerline)
    filtered = notch_filter(raw_signal, 50, fs)
    filtered = notch_filter(filtered, 60, fs)
    
    # 2. Aggressive Bandpass (0.5 - 35 Hz)
    # We lower the highcut slightly to 35Hz to kill more muscle noise
    filtered = bandpass_filter(filtered, 0.5, 35, fs, order=6)
    
    # 3. Moving Average Smoothing (20ms window)
    # This helps eliminate the 'thick band' of noise seen in the user's data
    filtered = smooth_signal(filtered, window_ms=20, fs=fs)
    
    # 4. Final Baseline Correction
    filtered = filtered - np.mean(filtered)
    
    # 5. Normalization (Z-score)
    normalized = (filtered - np.mean(filtered)) / (np.std(filtered) + 1e-8)
    
    return normalized