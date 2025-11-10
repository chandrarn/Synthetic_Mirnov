# FNO Predictor Refactoring Summary

## Overview
Successfully refactored the FNO predictor code into two focused modules:
1. **`data_caching.py`** - Data extraction and caching utilities
2. **`fno_predictor.py`** - FNO neural network architecture and training

## Files Created/Modified

### 1. `data_caching.py` (NEW - 380 lines)
**Purpose**: Handles all data extraction, preprocessing, and caching operations.

**Key Components**:
- `gen_sensor_ordering()` - Sorts sensors consistently (BPXX_ABK/GHK → BPXT_ABK/GHK → BP_AA_BOT → BP_AA_TOP)
- `CacheConfig` - Dataclass for configuration parameters
- `_open_datasets()` - Loads NetCDF files from directory
- `_extract_sensor_names()` - Extracts base sensor names from datasets
- `_define_mode_regions_time()` - Defines frequency bands around mode frequencies
- `_avg_band_per_sensor()` - **Modified as requested**: Finds frequency with max average real value across sensors/time, then averages real/imag at that frequency per sensor over time
- `_get_mode_labels()` - Extracts mode numbers (m or n) from dataset attributes
- `_maybe_geometry_from_bp_k()` - Optionally loads theta/phi geometry from C-Mod bp_k data
- `cache_training_dataset()` - Main caching function that processes datasets and saves to NPZ
- `build_or_load_cached_dataset()` - High-level interface (checks cache, loads or builds)

**Returns**: `(X_ri, y, sensor_names, theta, phi)` where:
- `X_ri`: (N, S, 2) array of [real, imag] features per sensor
- `y`: (N,) array of mode labels
- `sensor_names`: (S,) sorted sensor names
- `theta`, `phi`: (S,) optional geometry arrays

### 2. `fno_predictor.py` (NEW - 338 lines)
**Purpose**: Focuses exclusively on FNO neural network architecture and training.

**Key Components**:
- `SensorFourChannelDataset` - PyTorch Dataset wrapper for 4-channel [real, imag, theta, phi] input
  - Supports masking (randomly zero out sensors) and noise augmentation during training
- `SpectralConv1d` - 1D Fourier layer with learnable complex weights in spectral domain
- `FNOBlock1d` - FNO block combining spectral conv + pointwise conv + BatchNorm + activation
- `FNO1dClassifier` - Complete FNO model:
  - Lift: Conv1d to project 4 channels → width
  - Blocks: Stack of FNOBlock1d layers
  - Pool: AdaptiveAvgPool1d over sensors
  - Head: MLP classifier with dropout
- `TrainConfig` - Dataclass for training hyperparameters
- `train_fno_classifier()` - Complete training loop with:
  - Train/validation split
  - Early stopping with patience
  - Learning rate scheduling (ReduceLROnPlateau)
  - Validation accuracy tracking
  - Confusion matrix plotting
  - AUC computation
- `example_usage()` - End-to-end example demonstrating data loading → training → model saving

### 3. `fno_predictor_old.py` (BACKUP)
Original file backed up for reference.

### 4. `test_refactor.py` (TEST SCRIPT)
Quick validation script that:
- Imports both modules
- Tests sensor ordering function
- Instantiates and tests FNO model forward pass
- Verifies refactoring is successful

## Key Improvements

### Code Organization
- **Separation of concerns**: Data handling vs. neural network code completely separated
- **Cleaner imports**: `fno_predictor.py` only needs to import from `data_caching`
- **Focused files**: Each file has a single clear responsibility

### Maintainability
- **Easier to modify**: Data preprocessing changes don't touch neural network code
- **Better testing**: Can test data caching and model separately
- **Clearer documentation**: Each module has focused docstrings

### Data Processing Enhancement
- **Modified `_avg_band_per_sensor()`**: Now finds the single frequency bin with maximum average real value across all sensors (at given timepoint), then returns per-sensor time-averaged real/imag at that frequency
- **Consistent sensor ordering**: `gen_sensor_ordering()` ensures deterministic FFT operations in FNO

### Neural Network Features
- **4-channel input**: [real, imag, theta, phi] per sensor
- **Spectral operations**: True Fourier Neural Operator over sorted sensor axis
- **Data augmentation**: Sensor masking + noise during training
- **Robust training**: Early stopping, LR scheduling, validation tracking

## Usage

### 1. Cache dataset (one-time, fast subsequent loads)
```python
from data_caching import build_or_load_cached_dataset

X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="/path/to/netcdf/files/",
    out_path="/path/to/cache.npz",
    use_mode='n',  # or 'm'
    include_geometry=True,
    geometry_shot=1160714026,
    num_timepoints=-1,  # -1 for all
    freq_tolerance=0.1,
    n_datasets=-1  # -1 for all
)
```

### 2. Train FNO classifier
```python
from fno_predictor import train_fno_classifier, TrainConfig

model, le, val_acc, auc = train_fno_classifier(
    X_ri, y, theta=theta, phi=phi,
    cfg=TrainConfig(
        batch_size=64,
        n_epochs=300,
        patience=30,
        modes=24,
        width=192,
        depth=4,
        dropout=0.3
    ),
    plot_confusion=True,
    class_names=None  # Will use le.classes_
)
```

### 3. Save trained model
```python
import torch

torch.save({
    'state_dict': model.state_dict(),
    'classes': le.classes_.tolist(),
    'sensor_names': sensor_names,
    'theta': theta,
    'phi': phi
}, 'fno_classifier_n.pth')
```

## Verification

Test ran successfully (data_caching module verified):
```
✓ data_caching module imported successfully
  - build_or_load_cached_dataset
  - CacheConfig
  - gen_sensor_ordering
✓ Sensor ordering test passed
```

FNO module requires PyTorch (expected; install via `pip install torch` when ready to train).

## Next Steps

1. **Install PyTorch** if training: `pip install torch torchvision`
2. **Test caching**: Run data caching on your NetCDF datasets
3. **Train model**: Use cached data to train FNO classifier
4. **Tune hyperparameters**: Adjust `TrainConfig` for your specific dataset

## File Locations

```
/home/rianc/Documents/Synthetic_Mirnov/classifier_regressor/
├── data_caching.py          # NEW: Data extraction & caching
├── fno_predictor.py         # NEW: FNO neural network (clean)
├── fno_predictor_old.py     # BACKUP: Original version
└── test_refactor.py         # TEST: Verification script
```
