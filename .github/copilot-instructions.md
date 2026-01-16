# Synthetic Mirnov AI Coding Agent Instructions

## Project Overview
Python codebase for generating synthetic magnetic (Mirnov) signals for fusion plasma MHD mode analysis using OpenFUSIONToolkit's ThinCurr electromagnetic solver. Pipeline: generate synthetic sensor data → create spectrograms → train ML models (FNO/CNN) to classify mode numbers.

## Critical External Dependencies

**Required environment variables:**
- `OFT_ROOTPATH`: Path to OpenFUSIONToolkit build directory (e.g., `/home/user/OpenFUSIONToolkit/build_release/python/`)
- `TARS_ROOTPATH`: Path to TARS equilibrium field tool (filament generation)

**Key external tools:**
- OpenFUSIONToolkit (ThinCurr solver) - electromagnetic simulation engine
- TARS - field line tracing for plasma filament generation
- MDSplus/mdsthin - C-Mod experimental data access (optional)

**Installation note:** No `requirements.txt` exists. Infer dependencies from imports. Core: numpy, xarray, torch, sklearn, matplotlib, scipy, freeqdsk, cv2, pyvista.

## Architecture & Data Flows

### Major Components (4 subsystems)

**1. Signal Generation (`signal_generation/`)** 
- Entry: `Synthetic_Mirnov.py::gen_synthetic_Mirnov()` - main workflow orchestrator
- Generates synthetic Mirnov coil signals via ThinCurr finite element solver
- Key flow: gEQDSK equilibrium → field line traced filaments → ThinCurr mesh → sensor signals
- Outputs: JSON time series + HDF5 history files

**2. Synthetic Mirnov Generation Tool (Release) (`synthetic_mirnov_generation_tool_release/`)**
- Production-ready refactored version of signal generation
- Entry: `generate_synthetic_mirnov_signals.py::thincurr_synthetic_mirnov_signal()`
- Returns xarray Datasets with real/imag probe signals in [T/s]
- Modular: `prep_ThinCurr_input.py` (filaments/currents), `run_ThinCurr_model.py` (solver)

**3. ML Classifier/Regressor (`classifier_regressor/`)**
- **Data pipeline**: `data_caching.py` - extracts features from NetCDF spectrograms → caches to NPZ
  - Critical function: `_avg_band_per_sensor()` - finds peak frequency in mode band, averages real/imag per sensor over time
  - Sensor ordering: `gen_sensor_ordering()` - deterministic sort: BPXX_ABK/GHK → BPXT_ABK/GHK → BP_AA_BOT → BP_AA_TOP
- **FNO model**: `fno_predictor.py` - Fourier Neural Operator for mode number classification
  - 4-channel input per sensor: [real, imag, theta, phi] geometry
  - Spectral convolution over sorted sensor axis (1D FNO)
  - See `REFACTORING_SUMMARY.md` for data/model separation rationale
- **CNN model**: `cnn_predictor.py` - alternative convolutional architecture

**4. C-Mod Data Access (`C-Mod/`)**
- `get_Cmod_Data.py` - MDSplus interface to Alcator C-Mod experimental tokamak data
- Geometry files: `C_Mod_Mirnov_Geometry*.json`, `C_Mod_BP_Geometry*.json`
- Used for real data comparison and geometry injection into synthetic datasets

### Critical Workflows

**Generating synthetic training data:**
```bash
cd signal_generation/
python batch_run_synthetic_spectrogram.py  # Batch generation → NetCDF spectrograms
```
- Calls `gen_synthetic_Mirnov()` per mode → runs ThinCurr → computes spectrograms
- Output: `data_output/*.nc` xarray datasets with dimensions (sensor, frequency, time)
- Mode info stored in attrs: `mode_m`, `mode_n`, time-varying frequency in `F_Mode_*` vars

**Training FNO classifier:**
```python
from data_caching import build_or_load_cached_dataset
from fno_predictor import train_fno_classifier, TrainConfig

# Cache dataset (one-time, fast reload)
X_ri, y, sensor_names, theta, phi = build_or_load_cached_dataset(
    data_dir="path/to/netcdf/", out_path="cached.npz", use_mode='n'
)

# Train model
model, le, val_acc, auc = train_fno_classifier(X_ri, y, theta=theta, phi=phi)
```

## Project-Specific Patterns

### Import conventions - Header file pattern
**Almost every module imports from domain-specific header files:**
- `from header_signal_generation import ...` (signal_generation/)
- `from header_Cmod import ...` (C-Mod/)
- `from header_synthetic_mirnov_generation import ...` (synthetic_mirnov_generation_tool_release/)

Headers centralize imports and configure plotting (LaTeX rendering, TkAgg backend). Check headers first for available utilities.

### Path manipulation
Frequent relative imports via `sys.path.append('../C-Mod/')`, `sys.path.append('../signal_analysis/')`. Code assumes execution from subdirectories (not workspace root).

### File formats & naming
- **gEQDSK files**: Equilibrium data (e.g., `g1051202011.1000`) - read via `freeqdsk.geqdsk`
- **ThinCurr mesh**: `.h5` files (e.g., `C_Mod_ThinCurr_Combined-homology.h5`)
- **Sensor locations**: `.loc` files in `input_data/floops_<sensor_set>.loc`
- **Spectrograms**: NetCDF `.nc` with `{sensor_name}_real` and `{sensor_name}_imag` variables
- **Cached training data**: `.npz` files with keys `X_ri`, `y`, `sensor_names`, `theta`, `phi`

### Data structures
- **Mode parameters**: Dict with keys `{'m', 'n', 'r', 'R', 'n_pts', 'm_pts', 'f', 'dt', 'T', 'periods', 'n_threads', 'I'}`
  - `m`, `n`: poloidal/toroidal mode numbers (can be lists for multi-mode)
  - `f`: frequency (can be time-varying array)
  - `n_pts`, `m_pts`: filament discretization resolution
- **Sensor naming**: Case-insensitive regex patterns (`BPXX_ABK`, `BP_AA_TOP`) - use `gen_sensor_ordering()` for consistency

### Plotting & visualization
- All plotting uses matplotlib with LaTeX rendering enabled (`rc('text', usetex=True)`)
- TkAgg backend enforced (interactive plots)
- PyVista for 3D mesh/geometry visualization
- Common pattern: `plt.close('FigureName'); plt.figure('FigureName')` to reuse figure windows

## Key Files Reference

- **Main workflows**: `signal_generation/Synthetic_Mirnov.py`, `signal_generation/batch_run_synthetic_spectrogram.py`
- **Production tool**: `synthetic_mirnov_generation_tool_release/generate_synthetic_mirnov_signals.py`
- **ML pipeline**: `classifier_regressor/data_caching.py` → `classifier_regressor/fno_predictor.py`
- **Geometry**: `signal_generation/gen_MAGX_Coords.py`, `C-Mod/mirnov_Probe_Geometry.py`
- **Filament generation**: `signal_generation/geqdsk_filament_generator.py`, `synthetic_mirnov_generation_tool_release/prep_ThinCurr_input.py`

## Common Pitfalls

1. **Missing environment variables**: Check `OFT_ROOTPATH` and `TARS_ROOTPATH` are set before running anything
2. **Path dependencies**: Code expects execution from specific subdirectories - check `sys.path.append` calls
3. **Data file locations**: Input files often in `input_data/`, `../C-Mod/`, or hardcoded absolute paths - search for file references
4. **Sensor ordering consistency**: Always use `gen_sensor_ordering()` when working with multiple sensors/datasets
5. **Multi-mode complexity**: `m`, `n`, `f`, `I` can be scalars OR lists - check type before processing

## Testing & Validation

No formal test suite. Manual testing via:
- `classifier_regressor/test_refactor.py` - validates data_caching and FNO imports
- Main modules have `if __name__ == "__main__":` blocks with example usage

## Debugging Tips

- Enable debug flags: Most functions have `debug=True` parameter for verbose output
- Check OFT initialization: `oft_env = OFT_env(nthreads=n)` - may abort if misconfigured
- Plotting: Use `doPlot=True` parameters to visualize intermediate results (meshes, filaments, sensors)
- Data inspection: xarray Datasets have `.attrs` with metadata - always check for mode info
