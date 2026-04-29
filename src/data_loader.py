import pandas as pd
import numpy as np
import scipy.io
import wfdb
import os
import tempfile

def load_signal_data(uploaded_file):
    """
    Loads signal data from CSV, MAT, or DAT (WFDB) formats.
    Returns: (amplitude_array, sampling_frequency)
    """
    filename = uploaded_file.name
    ext = os.path.splitext(filename)[1].lower()
    
    # We use a temporary file because some libraries (like wfdb) need a real path
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        if ext == '.csv':
            df = pd.read_csv(tmp_path)
            # Try to find common amplitude column names
            col = next((c for c in df.columns if c.lower() in ['amplitude', 'val', 'signal', 'ecg', 'lead_i']), df.columns[0])
            amplitude = df[col].values
            # Try to find Fs from columns or header? For CSV we usually rely on user input
            return amplitude, None

        elif ext == '.mat':
            mat_data = scipy.io.loadmat(tmp_path)
            # Find the first non-header key that contains an array
            for key in mat_data:
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    amplitude = mat_data[key].flatten()
                    return amplitude, None
            raise ValueError("No valid signal array found in MAT file.")

        elif ext == '.dat':
            # For .dat, wfdb usually expects a header file (.hea) in the same directory.
            # If only .dat is provided, it might be a raw binary. 
            # Here we try wfdb rdrecord assuming the user might have uploaded or we can parse it.
            # NOTE: rdrecord requires the file name without extension and assumes .hea exists.
            record_name = os.path.splitext(tmp_path)[0]
            try:
                record = wfdb.rdrecord(record_name)
                amplitude = record.p_signal[:, 0] # Take first channel
                return amplitude, record.fs
            except Exception:
                # Fallback: Assume raw 16-bit PCM if wfdb fails
                amplitude = np.fromfile(tmp_path, dtype=np.int16)
                return amplitude, None
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
            
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def detect_fs(data, default_fs=500):
    """
    Placeholder for any Fs detection logic.
    """
    return default_fs
