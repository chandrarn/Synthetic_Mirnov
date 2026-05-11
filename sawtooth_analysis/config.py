"""
Configuration module for sawtooth precursor analysis.

Define analysis parameters for each shot:
- time_range: (start_s, end_s) in seconds
- freq_range: (start_hz, end_hz) in Hz
- target_mode: (n, m) tuple or list of (n, m) tuples to merge ambiguous labels
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SawtoothAnalysisConfig:
    """Configuration for a single shot analysis."""
    
    shot: int
    """Shot number"""
    
    time_range: Tuple[float, float]
    """Time window to search for bursts [start_s, end_s]"""
    
    freq_range: Tuple[float, float]
    """Frequency window to search for bursts [start_hz, end_hz]"""
    
    target_mode: Tuple[int, int] | tuple[Tuple[int, int], ...] | list[Tuple[int, int]]
    """Target mode(s) (n, m) for sawtooth precursor identification"""
    
    eq_time_idx: int | None = None
    """Equilibrium time index. If None, uses midpoint of shot."""
    
    chisq_threshold: float = 0.7
    """Chi-squared threshold for mode identification quality"""

    min_area_points: int = 5
    """Minimum connected points required to keep a burst"""

    min_time_span_s: float = 0.0
    """Minimum burst duration in seconds"""

    min_freq_span_hz: float = 0.0
    """Minimum burst bandwidth in Hz"""

    line: int = 2
    """Spectral line of interest (default: 2 = He-Like Z)"""

    tht: int = 0
    """THACO Analysis Tree number (default: 0 = None)"""

    max_hirexsr_omega_err_khz: float | None = 10.0
    """Maximum allowed HIREXSR q=1 rotation uncertainty in kHz (None disables cut)"""

    min_hirexsr_snr: float | None = None
    """Minimum required q=1 rotation SNR = |f_tor|/sigma_f (None disables cut)"""

    plot_diamagnetic_drifts: bool = True
    """Whether to generate the q=1 diamagnetic/rotation time-trace plot"""


# Example shot configurations
SHOT_CONFIGS = {
    1120927023: SawtoothAnalysisConfig(
        shot=1120927023,
        time_range=(0.90, 1.30),  # From your example output
        freq_range=(0, 50e3),      # 0-50 kHz
        target_mode=(-1, -1),      # n=-1 sawtooth precursor
        eq_time_idx=11000,
        chisq_threshold=0.70,
    ),
    # Add more shots as needed
}


def get_config(shot: int) -> SawtoothAnalysisConfig:
    """Retrieve configuration for a specific shot."""
    if shot not in SHOT_CONFIGS:
        raise ValueError(
            f"No configuration found for shot {shot}. "
            f"Available: {list(SHOT_CONFIGS.keys())}"
        )
    return SHOT_CONFIGS[shot]


def register_config(config: SawtoothAnalysisConfig) -> None:
    """Register a new shot configuration."""
    SHOT_CONFIGS[config.shot] = config
