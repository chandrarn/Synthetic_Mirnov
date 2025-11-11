# Dataset Visualization Feature - Summary

## Changes Made

Successfully incorporated the visualization code from `linear_fit_phase.py` (lines 317-350) into `data_caching.py` to enable visual inspection of dataset files during the caching process.

## New Features

### 1. `visualize_dataset_regions()` Function
A new function that creates a diagnostic plot showing:
- **Background**: Real spectrogram for a selected sensor
- **Colored rectangles**: Frequency-time regions used for feature extraction
- **Legend**: Mode numbers (m/n) for each region
- **Customizable**: Sensor selection, frequency limits, save path

**Location**: `classifier_regressor/data_caching.py` (lines ~265-350)

### 2. Enhanced `CacheConfig` Dataclass
Added new optional configuration parameters:
- `visualize_first: bool = False` - Enable/disable visualization
- `viz_save_path: Optional[str] = None` - Path to save plots
- `viz_sensor: str = 'BP01_ABK'` - Sensor to visualize
- `viz_freq_lim: Optional[List[float]] = None` - Frequency limits [f_min, f_max] in kHz

### 3. Updated `build_or_load_cached_dataset()` 
Now accepts visualization parameters:
```python
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="path/to/netcdf/",
    out_path="cached.npz",
    visualize_first=True,  # NEW: Enable visualization
    viz_save_path='../output_plots/',  # NEW: Save location
    viz_sensor='BP01_ABK',  # NEW: Sensor selection
    viz_freq_lim=[0, 300]  # NEW: Frequency limits (kHz)
)
```

### 4. Automatic Sensor Fallback
If the specified `viz_sensor` doesn't exist in the dataset, the function:
1. Lists available sensors
2. Automatically selects the first available sensor
3. Continues with visualization (no crash)

## Usage Examples

### Basic Usage (No Visualization)
```python
from data_caching import build_or_load_cached_dataset

X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/",
    out_path="cached_data.npz"
)
```

### With Visualization (Recommended for First Run)
```python
from data_caching import build_or_load_cached_dataset

X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="../data_output/synthetic_spectrograms/",
    out_path="cached_data.npz",
    visualize_first=True,  # Shows diagnostic plot for first dataset
    viz_save_path='../output_plots/',  # Saves as PDF
    viz_sensor='BP01_ABK',
    viz_freq_lim=[0, 300]  # Limit y-axis to 0-300 kHz
)
```

### Test Script
A complete example is provided in:
```bash
python test_visualization.py
```

## What the Visualization Shows

The plot displays:
1. **X-axis**: Time (in milliseconds)
2. **Y-axis**: Frequency (in kilohertz)
3. **Background colormap**: Real component of spectrogram signal [T/s]
4. **Rectangles**: 
   - Each rectangle marks a frequency-time region used for training data extraction
   - Color indicates which mode (different m/n pairs have different colors)
   - Width = temporal sampling resolution
   - Height = frequency band around mode peak (determined by `freq_tolerance`)

## Benefits

1. **Quality Control**: Visually verify that frequency bands correctly track mode evolution
2. **Parameter Tuning**: Adjust `freq_tolerance` if bands are too wide/narrow
3. **Debugging**: Identify issues with mode frequency tracking or zero-frequency periods
4. **Documentation**: Generate publication-ready figures showing data extraction regions

## Integration with Existing Workflow

The visualization is **opt-in** and doesn't affect existing code:
- `visualize_first=False` (default) → No visualization, original behavior
- `visualize_first=True` → Visualization for first dataset only (minimal overhead)
- Visualization only runs during **new cache creation**, not when loading existing cache

## Files Modified

1. **`classifier_regressor/data_caching.py`**:
   - Added matplotlib imports
   - Added `visualize_dataset_regions()` function
   - Extended `CacheConfig` dataclass
   - Updated `build_or_load_cached_dataset()` signature
   - Modified `cache_training_dataset()` to call visualization

2. **`classifier_regressor/test_visualization.py`** (NEW):
   - Complete example demonstrating visualization feature

## Compatibility

- Fully backward compatible with existing code
- All visualization parameters are optional with sensible defaults
- No changes required to existing scripts unless visualization is desired
