"""Sawtooth precursor analysis package."""

__version__ = "0.1.0"
__author__ = "MHD Spectroscopy Analysis"

from .config import SawtoothAnalysisConfig, get_config, register_config
from .burst_detection import (
    SawtoothBurst,
    detect_bursts_connected_components,
    filter_bursts_by_region,
    print_burst_summary,
)
from .main import SawtoothAnalysisPipeline

__all__ = [
    "SawtoothAnalysisConfig",
    "get_config",
    "register_config",
    "SawtoothBurst",
    "detect_bursts_connected_components",
    "filter_bursts_by_region",
    "print_burst_summary",
    "SawtoothAnalysisPipeline",
]
