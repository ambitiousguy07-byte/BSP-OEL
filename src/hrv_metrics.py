import numpy as np
from scipy import signal, interpolate
from scipy.stats import entropy

def get_rr_intervals(r_peaks, fs):
    """
    Compute RR intervals in milliseconds and remove physiological and statistical outliers.
    """
    if len(r_peaks) < 2:
        return np.array([])
        
    rr_intervals = (np.diff(r_peaks) / fs) * 1000.0  # ms
    
    # 1. Physiological limits (300ms to 2000ms)
    valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
    
    # 2. Statistical / Percentage outlier removal (e.g., Malik's rule)
    # Remove intervals that differ by more than 20% from the previous one
    if len(valid_rr) > 5:
        # Simple median filter approach for outliers
        median_rr = np.median(valid_rr)
        # Keep intervals within 20% of the median or local mean
        valid_rr = valid_rr[np.abs(valid_rr - median_rr) < 0.2 * median_rr]
        
    return valid_rr

def analyze_time_domain(rr_intervals):
    """
    Advanced Statistical and Time-domain HRV metrics.
    """
    if len(rr_intervals) < 2:
        return {}
        
    diff_rr = np.diff(rr_intervals)
    
    # 1. Variability Measures
    sdnn = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(diff_rr**2))
    nn50 = np.sum(np.abs(diff_rr) > 50)
    pnn50 = (nn50 / len(diff_rr)) * 100 if len(diff_rr) > 0 else 0
    variance = np.var(rr_intervals)
    
    # 2. Central Tendency
    mean_rr = np.mean(rr_intervals)
    median_rr = np.median(rr_intervals)
    
    # 3. Distribution Shape
    from scipy.stats import skew, kurtosis
    rr_skew = skew(rr_intervals)
    rr_kurtosis = kurtosis(rr_intervals)
    
    # 4. Range and Percentiles
    max_rr = np.max(rr_intervals)
    min_rr = np.min(rr_intervals)
    rr_range = max_rr - min_rr
    q1 = np.percentile(rr_intervals, 25)
    q3 = np.percentile(rr_intervals, 75)
    iqr = q3 - q1
    
    return {
        "SDNN": sdnn,
        "RMSSD": rmssd,
        "pNN50": pnn50,
        "NN50": nn50,
        "Mean_RR": mean_rr,
        "Median_RR": median_rr,
        "Variance": variance,
        "Skewness": rr_skew,
        "Kurtosis": rr_kurtosis,
        "Max_RR": max_rr,
        "Min_RR": min_rr,
        "Range": rr_range,
        "Q1": q1,
        "Q3": q3,
        "IQR": iqr
    }

def analyze_frequency_domain(rr_intervals, fs_interp=4.0):
    """
    Frequency-domain HRV metrics using Welch's method.
    """
    # 1. Resample RR intervals to uniform grid (standard for PSD)
    time = np.cumsum(rr_intervals) / 1000.0
    time = time - time[0]
    
    f_interp = interpolate.interp1d(time, rr_intervals, kind='cubic')
    
    new_time = np.arange(0, time[-1], 1/fs_interp)
    rr_resampled = f_interp(new_time)
    
    # 2. Welch PSD
    f, psd = signal.welch(rr_resampled, fs=fs_interp, nperseg=min(len(rr_resampled), 256))
    
    # 3. Define bands
    vlf_band = (0.0033, 0.04)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    
    vlf = np.trapezoid(psd[(f >= vlf_band[0]) & (f < vlf_band[1])], f[(f >= vlf_band[0]) & (f < vlf_band[1])])
    lf = np.trapezoid(psd[(f >= lf_band[0]) & (f < lf_band[1])], f[(f >= lf_band[0]) & (f < lf_band[1])])
    hf = np.trapezoid(psd[(f >= hf_band[0]) & (f < hf_band[1])], f[(f >= hf_band[0]) & (f < hf_band[1])])
    
    lf_hf_ratio = lf / hf if hf > 0 else 0
    
    return {
        "LF": lf,
        "HF": hf,
        "LF_HF_Ratio": lf_hf_ratio,
        "f": f,
        "psd": psd
    }

def analyze_nonlinear(rr_intervals):
    """
    Non-linear HRV metrics: Poincaré and Entropy.
    """
    if len(rr_intervals) < 10:
        return {"SD1": 0, "SD2": 0, "SampEn": 0, "ApEn": 0, "rr_n": [], "rr_n1": []}
        
    # Poincaré
    rr_n = rr_intervals[:-1]
    rr_n1 = rr_intervals[1:]
    
    diff_rr = np.diff(rr_intervals)
    sd1 = np.sqrt(np.std(diff_rr)**2 / 2)
    sd2 = np.sqrt(2 * np.std(rr_intervals)**2 - sd1**2)
    
    # Entropy Suite
    samp_en = calculate_sample_entropy(rr_intervals)
    ap_en = calculate_approx_entropy(rr_intervals)
    
    return {
        "SD1": sd1,
        "SD2": sd2,
        "SampEn": samp_en,
        "ApEn": ap_en,
        "rr_n": rr_n,
        "rr_n1": rr_n1
    }

def calculate_approx_entropy(data, m=2, r=None):
    """
    Calculates Approximate Entropy (ApEn).
    """
    if r is None:
        r = 0.2 * np.std(data)
    
    n = len(data)
    
    def _phi(m):
        x = np.array([data[i:i+m] for i in range(n-m+1)])
        C = []
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            C.append(np.sum(dist <= r) / (n-m+1))
        return np.mean(np.log(C))
    
    try:
        return abs(_phi(m) - _phi(m+1))
    except:
        return 0

def calculate_sample_entropy(data, m=2, r=None):
    """
    Calculates Sample Entropy.
    """
    if r is None:
        r = 0.2 * np.std(data)
    
    n = len(data)
    
    def _phi(m):
        x = np.array([data[i:i+m] for i in range(n-m+1)])
        # Use broadcasting for distance calculation
        C = 0
        for i in range(len(x)):
            dist = np.max(np.abs(x - x[i]), axis=1)
            C += np.sum(dist <= r) - 1 # exclude self
        return C / (n-m+1)
    
    phi_m = _phi(m)
    phi_m1 = _phi(m+1)
    
    if phi_m == 0 or phi_m1 == 0:
        return 0
        
    return -np.log(phi_m1 / phi_m)

def get_clinical_interpretation():
    """
    Returns clinical interpretation strings for HRV metrics.
    """
    return {
        "SDNN": "Reflects overall Heart Rate Variability and total autonomic activity. Low values (<50ms) indicate high stress or autonomic dysfunction.",
        "RMSSD": "Primary marker of parasympathetic (vagal) activity. Higher values suggest better recovery and lower stress.",
        "pNN50": "Closely correlated with RMSSD; reflects parasympathetic activity.",
        "LF": "Associated with both sympathetic and parasympathetic activity, often linked to baroreflex sensitivity.",
        "HF": "Reflects parasympathetic (vagal) activity and respiratory sinus arrhythmia (RSA).",
        "LF/HF": "Commonly used as an indicator of Sympathovagal Balance. High values suggest sympathetic dominance; low values suggest parasympathetic dominance.",
        "SD1": "Represents short-term HRV (parasympathetic activity).",
        "SD2": "Represents long-term HRV and overall variability.",
        "SampEn": "Measures signal complexity. Lower values indicate more regularity (often seen in pathological states or aging)."
    }
