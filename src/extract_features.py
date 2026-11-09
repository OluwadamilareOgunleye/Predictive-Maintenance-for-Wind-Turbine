"""
Feature Extraction Pipeline for Wind Turbine Predictive Maintenance
Extracts statistical vibration features (RMS, Peak, Crest Factor, Kurtosis, Skewness, etc.)
from NREL Gearbox vibration dataset (accelerometer channels AN3 - AN10).
"""

import os
import glob
import time
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.stats as stats

CHANNELS = ['AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8', 'AN9', 'AN10']
WINDOW_SIZE = 20000  # 20,000 samples = 0.5s at 40kHz sampling rate
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = (
    os.path.join(BASE_DIR, 'data', 'raw', 'nrel')
    if os.path.exists(os.path.join(BASE_DIR, 'data', 'raw', 'nrel'))
    else os.path.join(BASE_DIR, 'raw', 'nrel')
)
PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')

def extract_window_features(signal_window):
    """
    Extracts time-domain statistical indicators for a 1D vibration signal window.
    """
    mean_val = float(np.mean(signal_window))
    std_val = float(np.std(signal_window))
    rms_val = float(np.sqrt(np.mean(signal_window ** 2)))
    peak_val = float(np.max(np.abs(signal_window)))
    p2p_val = float(np.ptp(signal_window))
    abs_mean = float(np.mean(np.abs(signal_window)))
    
    # Ratios and higher-order moments
    crest_factor = peak_val / (rms_val + 1e-9)
    kurt_val = float(stats.kurtosis(signal_window))
    skew_val = float(stats.skew(signal_window))
    shape_factor = rms_val / (abs_mean + 1e-9)
    impulse_factor = peak_val / (abs_mean + 1e-9)
    
    return {
        'mean': mean_val,
        'std': std_val,
        'rms': rms_val,
        'peak': peak_val,
        'p2p': p2p_val,
        'crest_factor': crest_factor,
        'kurtosis': kurt_val,
        'skewness': skew_val,
        'shape_factor': shape_factor,
        'impulse_factor': impulse_factor
    }

def process_all_files():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    healthy_files = sorted(glob.glob(os.path.join(RAW_DIR, 'Healthy', '*.mat')))
    damaged_files = sorted(glob.glob(os.path.join(RAW_DIR, 'Damaged', '*.mat')))
    
    print(f"Found {len(healthy_files)} Healthy files and {len(damaged_files)} Damaged files.")
    
    all_records = []
    sample_signals = {}  # Store short raw snippets for web visualizer
    
    # Process healthy files (label = 0)
    for fpath in healthy_files:
        fname = os.path.basename(fpath).replace('.mat', '')
        t0 = time.time()
        mat = sio.loadmat(fpath, variable_names=CHANNELS)
        
        # Save sample snippet from H1
        if fname == 'H1' and 'healthy' not in sample_signals:
            sample_signals['healthy'] = {ch: mat[ch][:5000].flatten() for ch in CHANNELS}
        
        n_samples = mat[CHANNELS[0]].shape[0]
        n_windows = n_samples // WINDOW_SIZE
        
        for w in range(n_windows):
            start_idx = w * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            row = {
                'file_id': fname,
                'window_idx': w,
                'label': 0,
                'status': 'Healthy'
            }
            for ch in CHANNELS:
                window_data = mat[ch][start_idx:end_idx].flatten()
                feats = extract_window_features(window_data)
                for f_name, f_val in feats.items():
                    row[f'{ch}_{f_name}'] = f_val
            all_records.append(row)
        print(f"Processed {fname} (Healthy) - {n_windows} windows in {time.time()-t0:.2f}s")
        
    # Process damaged files (label = 1)
    for fpath in damaged_files:
        fname = os.path.basename(fpath).replace('.mat', '')
        t0 = time.time()
        mat = sio.loadmat(fpath, variable_names=CHANNELS)
        
        # Save sample snippet from D1
        if fname == 'D1' and 'damaged' not in sample_signals:
            sample_signals['damaged'] = {ch: mat[ch][:5000].flatten() for ch in CHANNELS}
            
        n_samples = mat[CHANNELS[0]].shape[0]
        n_windows = n_samples // WINDOW_SIZE
        
        for w in range(n_windows):
            start_idx = w * WINDOW_SIZE
            end_idx = start_idx + WINDOW_SIZE
            row = {
                'file_id': fname,
                'window_idx': w,
                'label': 1,
                'status': 'Damaged'
            }
            for ch in CHANNELS:
                window_data = mat[ch][start_idx:end_idx].flatten()
                feats = extract_window_features(window_data)
                for f_name, f_val in feats.items():
                    row[f'{ch}_{f_name}'] = f_val
            all_records.append(row)
        print(f"Processed {fname} (Damaged) - {n_windows} windows in {time.time()-t0:.2f}s")

    df = pd.DataFrame(all_records)
    print(f"\nExtracted total dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['status'].value_counts()}")
    
    # Save full dataset
    out_csv = os.path.join(PROCESSED_DIR, 'wind_turbine_features.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved complete features to: {out_csv}")
    
    # Save raw sample snippets for visualization
    sample_npz = os.path.join(PROCESSED_DIR, 'sample_vibration_signals.npz')
    np.savez_compressed(
        sample_npz,
        h_an3=sample_signals['healthy']['AN3'],
        h_an4=sample_signals['healthy']['AN4'],
        h_an8=sample_signals['healthy']['AN8'],
        h_an10=sample_signals['healthy']['AN10'],
        d_an3=sample_signals['damaged']['AN3'],
        d_an4=sample_signals['damaged']['AN4'],
        d_an8=sample_signals['damaged']['AN8'],
        d_an10=sample_signals['damaged']['AN10'],
    )
    print(f"Saved sample waveforms to: {sample_npz}")
    
    # Perform clean train/test split:
    # We hold out H9, H10 and D9, D10 for test to ensure pure out-of-session evaluation
    test_files = ['H9', 'H10', 'D9', 'D10']
    train_df = df[~df['file_id'].isin(test_files)].copy()
    test_df = df[df['file_id'].isin(test_files)].copy()
    
    train_csv = os.path.join(PROCESSED_DIR, 'train_features.csv')
    test_csv = os.path.join(PROCESSED_DIR, 'test_features.csv')
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Train set: {train_df.shape} (Files: {train_df['file_id'].unique().tolist()})")
    print(f"Test set:  {test_df.shape} (Files: {test_df['file_id'].unique().tolist()})")
    print("Feature extraction pipeline complete!")

if __name__ == '__main__':
    process_all_files()
