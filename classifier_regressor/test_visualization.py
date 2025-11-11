#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to demonstrate the dataset visualization feature in data_caching.py

This script shows how to visualize the frequency-time regions used for data extraction
from xarray NetCDF spectrogram files.
"""

from data_caching import build_or_load_cached_dataset

# Example 1: Load/cache dataset with visualization enabled
print("Example: Caching dataset with visualization of first dataset regions\n")

# Specify your data directory containing NetCDF files
data_dir = "../data_output/synthetic_spectrograms/low_m-n_testing/"
cache_path = "../data_output/cached/test_cached_with_viz.npz"

# Build or load cached dataset with visualization
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir=data_dir,
    out_path=cache_path,
    use_mode='n',
    include_geometry=True,
    geometry_shot=1160714026,
    num_timepoints=20,  # Sample 20 timepoints per dataset
    freq_tolerance=0.1,
    n_datasets=1,  # Load only first dataset for quick test
    
    # Visualization options
    visualize_first=True,  # Enable visualization of first dataset
    viz_save_path='../output_plots/',  # Save visualization plots here
    viz_sensor='BP01_ABK',  # Sensor to visualize (will auto-select if not found)
    viz_freq_lim=[0, 300]  # Frequency limits in kHz for y-axis
)

print(f"\nDataset loaded successfully!")
print(f"  X_ri shape: {X_ri.shape}")
print(f"  y shape: {y.shape}")
print(f"  Number of sensors: {len(sensor_names)}")
print(f"  Unique mode numbers: {sorted(set(y))}")

# The visualization plot shows:
# - Background: Real spectrogram for one sensor
# - Colored rectangles: Frequency-time regions used for feature extraction
# - Each color represents a different mode (m/n pair)
# - Rectangle width = time sampling point
# - Rectangle height = frequency band around mode frequency
