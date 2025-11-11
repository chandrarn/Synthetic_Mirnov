#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick examples for using the dataset visualization feature
"""

from data_caching import build_or_load_cached_dataset

# ============================================================================
# Example 1: Standard caching (no visualization)
# ============================================================================
print("Example 1: Standard caching without visualization")
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/",
    out_path="cached_standard.npz",
    use_mode='n'
)
print(f"Loaded {len(y)} samples with {len(sensor_names)} sensors\n")


# ============================================================================
# Example 2: First-time caching with visualization (recommended)
# ============================================================================
print("Example 2: First-time caching with diagnostic visualization")
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/low_m-n_testing/",
    out_path="cached_with_viz.npz",
    use_mode='n',
    num_timepoints=20,
    freq_tolerance=0.1,
    
    # Enable visualization
    visualize_first=True,
    viz_save_path='../output_plots/',
    viz_sensor='BP01_ABK',
    viz_freq_lim=[0, 300]  # Focus on 0-300 kHz range
)
print(f"Visualization saved to ../output_plots/Debug_BP01_ABK_Real_Spectrogram.pdf\n")


# ============================================================================
# Example 3: Quick inspection of different sensors
# ============================================================================
print("Example 3: Visualize different sensor")
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/",
    out_path="cached_sensor_test.npz",
    n_datasets=1,  # Just first dataset for quick test
    
    visualize_first=True,
    viz_sensor='BP_AA_BOT',  # Different sensor
    viz_save_path='../output_plots/'
)


# ============================================================================
# Example 4: Full frequency range visualization
# ============================================================================
print("Example 4: Full frequency range (no y-axis limits)")
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/",
    out_path="cached_full_freq.npz",
    n_datasets=1,
    
    visualize_first=True,
    viz_freq_lim=None  # Show full frequency range
)


# ============================================================================
# Example 5: Testing different frequency tolerance
# ============================================================================
print("Example 5: Testing wider frequency bands")
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/",
    out_path="cached_wide_bands.npz",
    freq_tolerance=0.2,  # 20% tolerance (wider bands)
    n_datasets=1,
    
    visualize_first=True,
    viz_save_path='../output_plots/'
)
print("Check visualization to see if wider bands capture mode better\n")


# ============================================================================
# Notes:
# ============================================================================
# - Visualization only runs on FIRST dataset when visualize_first=True
# - If sensor not found, will auto-select first available sensor
# - Rectangles show frequency-time regions used for feature extraction
# - Each color = different mode (m/n pair)
# - Useful for debugging freq_tolerance, num_timepoints settings
