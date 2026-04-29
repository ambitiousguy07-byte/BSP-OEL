import numpy as np
import pandas as pd
import wfdb
import os

def generate_synthetic_ecg(duration=60, fs=500, use_real_mitdb=True):
    """
    Provides real clinical ECG data from MIT-BIH or a high-fidelity synthetic model.
    """
    if use_real_mitdb:
        try:
            # Download a small segment of MIT-BIH record 100 (classic real ECG)
            # We take a 1-minute segment
            record = wfdb.rdrecord('100', pb_dir='mitdb', sampto=int(duration * 360))
            # Resample to the requested Fs if necessary (MIT-BIH is 360Hz)
            amplitude = record.p_signal[:, 0]
            # Standardize and Return
            return pd.DataFrame({
                'Amplitude': amplitude,
                'Fs': 360 # Inform the caller of the real Fs
            })
        except Exception as e:
            print(f"Internet fetch failed, using high-fidelity synthetic model: {e}")

    # HIGH-FIDELITY SYNTHETIC MODEL (Physiological Template)
    t = np.linspace(0, duration, duration * fs)
    ecg = np.zeros_like(t)
    
    # Template based on real QRS/P/T morphology
    def heartbeat_template(fs):
        t_hb = np.linspace(-0.4, 0.6, int(fs))
        # QRS complex (High frequency, high amplitude)
        qrs = 1.0 * np.exp(-((t_hb - 0.0)**2) / (0.01**2))
        # P wave (Low frequency, low amplitude)
        p_wave = 0.1 * np.exp(-((t_hb + 0.18)**2) / (0.03**2))
        # T wave (Medium frequency, medium amplitude)
        t_wave = 0.2 * np.exp(-((t_hb - 0.35)**2) / (0.05**2))
        return qrs + p_wave + t_wave
    
    template = heartbeat_template(fs)
    bpm = 72
    rr_mean = 60.0 / bpm
    
    # Generate beats with real HRV (slight variability)
    r_times = []
    curr = 0.5
    while curr < duration - 1:
        r_times.append(curr)
        curr += rr_mean + np.random.normal(0, 0.05)
    
    for rt in r_times:
        idx = int(rt * fs)
        start = max(0, idx - len(template)//2)
        end = min(len(ecg), start + len(template))
        ecg[start:end] += template[:end-start]
        
    # Add realistic noise and baseline wander
    noise = 0.03 * np.random.normal(size=len(t))
    baseline = 0.05 * np.sin(2 * np.pi * 0.1 * t)
    
    return pd.DataFrame({'Amplitude': ecg + noise + baseline, 'Fs': fs})
