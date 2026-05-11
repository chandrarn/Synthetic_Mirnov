# Sawtooth Precursor Analysis Pipeline

## Scientific Goal

Visualize the relationship between n=1 sawtooth precursor frequency and the rotational frequency of the plasma at the q=1 surface, to determine if diamagnetic drift corrections are necessary to make the data match observational constraints from HIREX-SR ion rotation measurements.

## Project Overview

This package automates the analysis pipeline:

1. **Chi-squared Mode Identification** (`fit_oneshot_noRecomputation`)
   - Computes chi-squared values for all MHD modes
   - Identifies best-fit mode at each time-frequency point
   - Filters by chi-squared threshold

2. **Burst Detection** (custom `burst_detection` module)
   - Uses connected-components clustering in 2D time-frequency space
   - Identifies contiguous regions corresponding to individual sawtooth bursts
   - Robust to noise while maintaining spatial coherence
   - Computes dominant frequency and time for each burst

3. **Diamagnetic Drift Computation** (`calc_diamagnetic_drift`)
   - Loads electron and ion temperature, density profiles
   - Computes diamagnetic drift frequencies at the q=1 surface
   - Extracts HIREX-SR predicted toroidal rotation at q=1
   - Returns rotation frequency as function of normalized flux

4. **Analysis & Visualization**
   - Creates scatter plot: Sawtooth frequency vs. q=1 rotation frequency
   - Generates burst-by-burst analysis table (CSV)
   - Optional diagnostic plots for each burst

## Directory Structure

```
sawtooth_analysis/
├── __init__.py                 # Package initialization
├── config.py                   # Shot configuration definitions
├── burst_detection.py          # Burst clustering algorithms
├── main.py                     # Main orchestration pipeline
├── README.md                   # This file
├── outputs/                    # Analysis results (created at runtime)
│   ├── shot_XXXXXXXXX/
│   │   ├── burst_analysis.csv
│   │   ├── chisq/              # Chi-squared computation results
│   │   └── plots/              # Diagnostic plots
│   └── sawtooth_vs_rotation_scatter.png
└── notebooks/                  # Jupyter notebooks for exploration (optional)
```

## Quick Start

### 1. Add Shot Configuration

Edit `config.py` to add your shot(s):

```python
from config import SawtoothAnalysisConfig, register_config

config = SawtoothAnalysisConfig(
    shot=1120927023,
    time_range=(0.90, 1.30),      # seconds
    freq_range=(0, 50e3),          # Hz
    target_mode=(-1, -1),          # (n, m) tuple
    eq_time_idx=11000,
    chisq_threshold=0.70,
)
register_config(config)
```

### 2. Run Analysis

```python
from main import SawtoothAnalysisPipeline

pipeline = SawtoothAnalysisPipeline(
    scratch_dir="/path/to/TARS/scratch",
    output_dir="./outputs",
    max_workers=4,
)

results = pipeline.run_analysis(shot=1120927023)
print(results)

pipeline.create_scatter_plot()
```

Or from command line:

```bash
cd /home/rianc/Documents/Synthetic_Mirnov/sawtooth_analysis
python main.py
```

## Key Classes

### SawtoothAnalysisConfig

Defines analysis parameters for a single shot:
- `shot`: Shot number
- `time_range`: (start, end) in seconds
- `freq_range`: (start, end) in Hz
- `target_mode`: (n, m) tuple for sawtooth precursor
- `eq_time_idx`: Equilibrium time index (optional)
- `chisq_threshold`: Quality threshold for mode identification

### SawtoothBurst

Represents a single identified burst:
- `burst_id`: Unique identifier
- `dominant_freq`: Mean frequency of burst
- `dominant_time`: Time centroid of burst
- `n_points`: Number of time-frequency grid points in burst
- `area_seconds_hz`: Approximate area in time-frequency space
- `freq_range`, `time_range`: Properties returning (min, max) tuples

### SawtoothAnalysisPipeline

Main orchestrator:
- `run_analysis(shot, config)`: Execute full pipeline
- `create_scatter_plot()`: Visualize all results

## Output Files

### `burst_analysis.csv`

Table with columns:
- `burst_id`: Burst identifier
- `dominant_time_s`: Burst time centroid
- `dominant_freq_hz`: Burst frequency (Hz)
- `omega_de_q1`: Electron diamagnetic drift (rad/s) at q=1
- `omega_di_q1`: Ion diamagnetic drift (rad/s) at q=1
- `omega_tor_q1`: HIREX-SR toroidal rotation (rad/s) at q=1
- Additional columns for statistical analysis

### `sawtooth_vs_rotation_scatter.png`

Scatter plot showing:
- X-axis: Sawtooth precursor frequency (kHz)
- Y-axis: q=1 rotation frequency (kHz)
- Multiple series for different drift corrections

## Burst Detection Algorithm

The pipeline uses **connected-components analysis** with Moore (8-point) connectivity to identify bursts:

1. Filter chi-squared map for target mode within threshold
2. Apply binary morphology to identify connected regions
3. Discard regions with fewer than `min_area_points` (default: 5) points
4. Compute statistics for each connected component
5. Filter final bursts to requested time-frequency window

### Why Connected Components?

- **Contiguity-aware**: Only connects adjacent time-frequency points (resists noise bleed)
- **No parameters to tune**: Unlike DBSCAN's eps/min_samples
- **Fast**: O(n) complexity
- **Robust**: Preserves spatial structure while eliminating scattered noise

Alternative algorithms (K-means, DBSCAN) can be added as new functions in `burst_detection.py`.

## Integration with Existing Code

This package wraps existing modules:

| Module | Function | Purpose |
|--------|----------|---------|
| `tars.workflow.fit_oneshot_noRecomputation` | `calc_chisq_all_modes_single_eq()` | Chi-squared computation |
| `tars.plotting.chisq_plots` | `plot_mode_chisq_diagnostic_grid()` | Visualization |
| `calc_diamagnetic_drift` | `compute_diamagnetic_drift_frequencies()` | Drift/rotation computation |

## Next Steps

1. ✅ Defined shot configuration(s) in `config.py`
2. ⚠️ Test burst detection on your example shot (1120927023)
3. ⚠️ Verify chi-squared mode identification
4. ⚠️ Validate diamagnetic drift calculation
5. ⚠️ Generate scatter plots and compare with expectations
6. ⚠️ Add diagnostic plots for each burst (optional)
7. ⚠️ Extend to multiple shots

## Troubleshooting

### ImportError for TARS modules

Ensure TARS and C-Mod directories are in Python path:

```python
import sys
sys.path.insert(0, "/path/to/TARS")
sys.path.insert(0, "/path/to/C-Mod")
```

### No bursts detected

- Check `time_range` and `freq_range` in config
- Verify `target_mode` matches actual data
- Lower `chisq_threshold` if too strict
- Inspect chi-squared diagnostic grid (`plot_mode_chisq_diagnostic_grid`)

### Drift computation fails

- Verify shot number has valid equilibrium and profile data
- Check time values are within available diagnostic windows
- Review profiles for missing data or gaps

## References

- Chi-squared fitting: [fit_oneshot_noRecomputation.py](../../../TARS/tars/workflow/)
- Diamagnetic drift: [calc_diamagnetic_drift.py](../C-Mod/)
- Connected components: scipy.ndimage.label

## Author Notes

- Connected components clustering selected for its spatial coherence properties
- Readily adaptable to different shot/mode combinations
- Can be extended with ML-based burst classification
- Planned: Comparison with alternative clustering methods
