"""
Burst detection algorithm for sawtooth precursor analysis.

Identifies contiguous regions in time-frequency space corresponding to
individual sawtooth precursor bursts using connectivity-based clustering.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
from scipy import ndimage


@dataclass
class SawtoothBurst:
    """Represents a single identified sawtooth precursor burst."""
    
    burst_id: int
    """Unique burst identifier"""
    
    time_indices: np.ndarray
    """Array of time indices belonging to this burst"""
    
    freq_indices: np.ndarray
    """Array of frequency indices belonging to this burst"""
    
    times: np.ndarray
    """Actual time values [s] for points in burst"""
    
    frequencies: np.ndarray
    """Actual frequency values [Hz] for points in burst"""
    
    dominant_freq: float
    """Dominant (mean) frequency for this burst [Hz]"""
    
    dominant_time: float
    """Time of burst centroid [s]"""
    
    n_points: int
    """Number of time-frequency points in burst"""
    
    area_seconds_hz: float
    """Approximate area in time-frequency space [s * Hz]"""
    
    @property
    def freq_range(self) -> Tuple[float, float]:
        """Return (min_freq, max_freq) for burst."""
        return (float(np.min(self.frequencies)), float(np.max(self.frequencies)))
    
    @property
    def time_range(self) -> Tuple[float, float]:
        """Return (min_time, max_time) for burst."""
        return (float(np.min(self.times)), float(np.max(self.times)))


def detect_bursts_connected_components(
    mode_map: np.ndarray,
    target_mode: int | Tuple[int, int],
    time: np.ndarray,
    frequency: np.ndarray,
    min_area_points: int = 5,
    connectivity: str = "moore",
) -> List[SawtoothBurst]:
    """
    Detect sawtooth precursor bursts using connected components analysis.
    
    Parameters
    ----------
    mode_map : np.ndarray
        2D array of shape (n_times, n_freqs) containing mode indices.
        Typically best_mode_idx_plot from chisq_plots.
    target_mode : int or tuple
        The mode index (or (n,m) tuple label) to identify as bursts.
    time : np.ndarray
        1D array of time coordinates [s]
    frequency : np.ndarray
        1D array of frequency coordinates [Hz]
    min_area_points : int
        Minimum number of connected points to be considered a burst.
    connectivity : str
        "moore" (8-connected) or "von_neumann" (4-connected)
    
    Returns
    -------
    List[SawtoothBurst]
        List of identified sawtooth bursts, sorted by time.
    """
    
    # Create binary mask for target mode
    # Handle both integer and (n,m) tuple mode labels
    if isinstance(target_mode, tuple):
        # If target_mode is (n,m), assume mode_map contains indices 
        # and we'll match by index into SEARCHED_MODES
        raise NotImplementedError(
            "Tuple mode matching requires SEARCHED_MODES reference. "
            "Pass integer mode index instead."
        )
    
    mask = mode_map == target_mode
    
    if not np.any(mask):
        return []  # No instances of target mode
    
    # Define connectivity structure
    if connectivity == "moore":
        structure = np.ones((3, 3), dtype=int)  # 8-connectivity
    else:  # von_neumann
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)  # 4-connectivity
    
    # Label connected components
    labeled_array, n_bursts = ndimage.label(mask, structure=structure)
    
    bursts = []
    for burst_id in range(1, n_bursts + 1):
        burst_mask = labeled_array == burst_id
        
        # Get indices of burst points
        time_indices, freq_indices = np.where(burst_mask)
        n_points = len(time_indices)
        
        # Filter by minimum area
        if n_points < min_area_points:
            continue
        
        # Extract actual values
        burst_times = time[time_indices]
        burst_freqs = frequency[freq_indices]
        
        # Compute statistics
        dominant_freq = np.mean(burst_freqs)
        dominant_time = np.mean(burst_times)
        
        # Estimate area as bounding box (conservative)
        time_span = np.max(burst_times) - np.min(burst_times)
        freq_span = np.max(burst_freqs) - np.min(burst_freqs)
        area = time_span * freq_span if time_span > 0 and freq_span > 0 else 0
        
        burst = SawtoothBurst(
            burst_id=len(bursts),  # Renumber starting from 0
            time_indices=time_indices,
            freq_indices=freq_indices,
            times=burst_times,
            frequencies=burst_freqs,
            dominant_freq=dominant_freq,
            dominant_time=dominant_time,
            n_points=n_points,
            area_seconds_hz=area,
        )
        bursts.append(burst)
    
    # Sort by time
    bursts.sort(key=lambda b: b.dominant_time)
    
    return bursts


def filter_bursts_by_region(
    bursts: List[SawtoothBurst],
    time_range: Tuple[float, float],
    freq_range: Tuple[float, float],
) -> List[SawtoothBurst]:
    """
    Filter bursts to keep only those within specified time-frequency region.
    
    A burst is kept if its centroid falls within both ranges.
    """
    filtered = []
    for burst in bursts:
        if (time_range[0] <= burst.dominant_time <= time_range[1] and
            freq_range[0] <= burst.dominant_freq <= freq_range[1]):
            filtered.append(burst)
    return filtered


def print_burst_summary(bursts: List[SawtoothBurst]) -> None:
    """Print summary table of detected bursts."""
    print(f"\nDetected {len(bursts)} bursts:\n")
    print(f"{'ID':<3} {'Time (s)':<12} {'Freq (kHz)':<14} {'N_pts':<7} {'Area':<12}")
    print("-" * 60)
    for burst in bursts:
        print(
            f"{burst.burst_id:<3} "
            f"{burst.dominant_time:<12.4f} "
            f"{burst.dominant_freq/1e3:<14.2f} "
            f"{burst.n_points:<7} "
            f"{burst.area_seconds_hz:<12.4e}"
        )
