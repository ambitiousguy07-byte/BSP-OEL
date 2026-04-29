import numpy as np
from scipy import signal

def pan_tompkins_detector(signal_data, fs):
    """
    Precision Pan-Tompkins with final refinement on the INPUT signal
    to ensure markers align perfectly with the displayed trace.
    """
    # 1. Internal Bandpass Filter for detection logic (5-15 Hz)
    nyq = 0.5 * fs
    low = 5 / nyq
    high = 15 / nyq
    b, a = signal.butter(1, [low, high], btype='band')
    detection_signal = signal.lfilter(b, a, signal_data)

    # Polarity for detection
    if np.abs(np.min(detection_signal)) > np.max(detection_signal):
        working_detection = -detection_signal
    else:
        working_detection = detection_signal

    # 2. Derivative/Squaring/Integration (Standard PT)
    derivative = np.diff(working_detection)
    squared = derivative ** 2
    window_size = int(0.200 * fs) 
    integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')

    # 3. Initial Peak Detection
    peaks, _ = signal.find_peaks(integrated, distance=int(0.25 * fs))
    
    # 4. Adaptive Thresholding
    r_peaks_coarse = []
    spki = np.max(integrated[:int(2*fs)]) * 0.25
    npki = np.mean(integrated[:int(2*fs)]) * 0.5
    
    for peak in peaks:
        threshold = npki + 0.25 * (spki - npki)
        if integrated[peak] > threshold:
            r_peaks_coarse.append(peak)
            spki = 0.125 * integrated[peak] + 0.875 * spki
        else:
            npki = 0.125 * integrated[peak] + 0.875 * npki

    # 5. ABSOLUTE PINPOINT REFINEMENT
    # IMPORTANT: We search for the max in the ORIGINAL signal_data 
    # to ensure the marker sits exactly on the tip of the plotted line.
    refined_peaks = []
    search_window = int(0.15 * fs) # 150ms search window
    
    # Check if the signal is inverted globally to find max or min
    # If the user's signal is mostly negative-going R-peaks, we find min
    is_inverted = np.abs(np.min(signal_data)) > np.max(signal_data)

    for p in r_peaks_coarse:
        start = max(0, p - search_window)
        end = min(len(signal_data), p + int(0.1 * fs))
        
        segment = signal_data[start:end]
        if len(segment) > 0:
            if is_inverted:
                refined_idx = start + np.argmin(segment)
            else:
                refined_idx = start + np.argmax(segment)
            refined_peaks.append(refined_idx)

    return np.unique(np.array(refined_peaks))

def calculate_hr(r_peaks, fs):
    if len(r_peaks) < 2: return np.array([]), np.array([])
    rr = np.diff(r_peaks) / fs
    hr = 60 / rr
    mask = (hr >= 30) & (hr <= 200)
    return (r_peaks[1:][mask] + r_peaks[:-1][mask]) / (2 * fs), hr[mask]
