#!/usr/bin/env python3
"""
Example usage of the sawtooth precursor analysis pipeline.

This script demonstrates:
1. Configuring a shot
2. Running the full analysis pipeline
3. Inspecting results
4. Creating visualizations
"""

import sys
from pathlib import Path
import numpy as np

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import SawtoothAnalysisConfig, register_config
from main import SawtoothAnalysisPipeline


def example_single_shot():
    """Example: Analyze a single shot."""
    
    print("\n" + "="*70)
    print("SAWTOOTH PRECURSOR ANALYSIS - EXAMPLE")
    print("="*70 + "\n")
    
    # Configuration
    scratch_dir = "/home/rianc/Documents/TARS/tars/scratch"
    output_dir = "/home/rianc/Documents/Synthetic_Mirnov/sawtooth_analysis/outputs"
    
    # Define analysis for shot 1120906030
    config = SawtoothAnalysisConfig(
        shot=1160826011,
        time_range=(0.5, 1.65),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=[(-3, -1)],  # Merge ambiguous labels
        eq_time_idx=11000,
        chisq_threshold=0.70,
        min_area_points=60,
        min_time_span_s=0.002,
        min_freq_span_hz=3000.0,
        line=2,
        tht=0,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        plot_diamagnetic_drifts=True,
        )
    

    print("Configuration:")
    print(f"  Shot: {config.shot}")
    print(f"  Time range: {config.time_range} s")
    print(f"  Freq range: {np.array(config.freq_range)/1e3} kHz")
    print(f"  Eq time index: {config.eq_time_idx}")
    print(f"  Target mode(s): {config.target_mode}")
    print(f"  Chi-sq threshold: {config.chisq_threshold}")
    print(f"  Min area points: {config.min_area_points}")
    print(f"  Min time span: {config.min_time_span_s} s")
    print(f"  Min freq span: {config.min_freq_span_hz/1e3} kHz")
    
    # Initialize pipeline
    pipeline = SawtoothAnalysisPipeline(
        scratch_dir=scratch_dir,
        output_dir=output_dir,
        use_multiprocessing=True,
        max_workers=4,
        debug=False,
    )
    
    # Run analysis
    print("\nRunning analysis pipeline...")
    try:
        results_df = pipeline.run_analysis(config.shot, config)
        
        # Display results
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nBurst Analysis Results:")
        print(results_df.to_string())
        
        # Create visualization
        print("\nGenerating scatter plot...")
        pipeline.create_scatter_plot(x_limits=(-15, 15), y_limits=None)
        
        print(f"\nResults saved to: {output_dir}")
        
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()


def example_multiple_shots():
    """Example: Analyze multiple shots sequentially."""
    
    # Define configurations for multiple shots
    configs = [
        SawtoothAnalysisConfig(
        shot=1120906030,
        time_range=(1.0, 1.50),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=(-2, -1),          # n=-1 sawtooth precursor
        eq_time_idx=12000,
        chisq_threshold=0.70,
        min_area_points=80,
        min_time_span_s=0.004,
        min_freq_span_hz=1000.0,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        line=2,
        tht=8,
        ),

        # Add more shot configurations as needed
        SawtoothAnalysisConfig(
        shot=1160718023,
        time_range=(0.5, 1.55),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=(3, 1),          # n=-1 sawtooth precursor
        eq_time_idx=11000,
        chisq_threshold=0.70,
        min_area_points=80,
        min_time_span_s=0.004,
        min_freq_span_hz=1000.0,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
    ),
    
        SawtoothAnalysisConfig(
        shot=1140729030,
        time_range=(0.5, 1.60),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=(2, 1),          # n=-1 sawtooth precursor
        eq_time_idx=12000,
        chisq_threshold=0.70,
        min_area_points=80,
        min_time_span_s=0.004,
        min_freq_span_hz=1000.0,
        line=2,
        tht=8,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        ),
    SawtoothAnalysisConfig(
        shot=1101014030,
        time_range=(0.5, 1.65),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=[(2, 1)],  # Merge ambiguous labels
        eq_time_idx=10000,
        chisq_threshold=0.70,
        min_area_points=60,
        min_time_span_s=0.002,
        min_freq_span_hz=3000.0,
        line=2,
        tht=1,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        plot_diamagnetic_drifts=True,
    ),

    SawtoothAnalysisConfig(
        shot=1101014029,
        time_range=(0.5, 1.65),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=[(2, 1)],  # Merge ambiguous labels
        eq_time_idx=11000,
        chisq_threshold=0.70,
        min_area_points=60,
        min_time_span_s=0.002,
        min_freq_span_hz=3000.0,
        line=2,
        tht=1,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        plot_diamagnetic_drifts=True,
        ),

    SawtoothAnalysisConfig(
        shot=1140729021,
        time_range=(0.5, 1.65),      # From your example plot
        freq_range=(6e3, 20e3),          # 0-20 kHz
        target_mode=[(2, 1)],  # Merge ambiguous labels
        eq_time_idx=11000,
        chisq_threshold=0.70,
        min_area_points=60,
        min_time_span_s=0.002,
        min_freq_span_hz=3000.0,
        line=2,
        tht=8,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        plot_diamagnetic_drifts=True,
        ),

        SawtoothAnalysisConfig(
        shot=1160920012,
        time_range=(0.5, 1.65),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=[(-3, -1)],  # Merge ambiguous labels
        eq_time_idx=11000,
        chisq_threshold=0.70,
        min_area_points=60,
        min_time_span_s=0.002,
        min_freq_span_hz=3000.0,
        line=2,
        tht=0,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        plot_diamagnetic_drifts=True,
        ),

        SawtoothAnalysisConfig(
        shot=1160826011,
        time_range=(0.5, 1.65),      # From your example plot
        freq_range=(0, 20e3),          # 0-20 kHz
        target_mode=[(-3, -1)],  # Merge ambiguous labels
        eq_time_idx=11000,
        chisq_threshold=0.70,
        min_area_points=60,
        min_time_span_s=0.002,
        min_freq_span_hz=3000.0,
        line=2,
        tht=0,
        max_hirexsr_omega_err_khz=10.0,
        min_hirexsr_snr=1.5,
        plot_diamagnetic_drifts=True,
        )
    
    ]
    
    # Initialize pipeline (once)
    pipeline = SawtoothAnalysisPipeline(
        scratch_dir="/home/rianc/Documents/TARS/tars/scratch",
        output_dir="/home/rianc/Documents/Synthetic_Mirnov/sawtooth_analysis/outputs",
        use_multiprocessing=False,  # Disable for sequential analysis
        max_workers=2,
    )
    
    # Analyze each shot
    all_results = []
    for config in configs:
        try:
            results_df = pipeline.run_analysis(config.shot, config)
            all_results.append(results_df)
        except Exception as e:
            print(f"Error analyzing shot {config.shot}: {e}")
            continue
    
    # Combined scatter plot
    if all_results:
        pipeline.create_scatter_plot(include_diamagnetic_drifts=False)


def example_custom_burst_detection():
    """Example: Use burst detection directly on pre-computed chi-squared data."""
    
    import numpy as np
    from burst_detection import (
        detect_bursts_connected_components,
        filter_bursts_by_region,
        print_burst_summary,
    )
    
    # Simulate chi-squared data
    n_times, n_freqs, n_modes = 100, 50, 16
    
    # Random mode identification (in practice, from chi-squared fit)
    mode_map = np.random.randint(0, n_modes, size=(n_times, n_freqs))
    
    # Add a synthetic burst (target_mode=5)
    burst_region = (slice(20, 40), slice(10, 20))
    mode_map[burst_region] = 5
    
    # Coordinates
    time = np.linspace(0.9, 1.3, n_times)
    frequency = np.linspace(0, 50e3, n_freqs)
    
    # Detect bursts
    bursts = detect_bursts_connected_components(
        mode_map=mode_map,
        target_mode=5,
        time=time,
        frequency=frequency,
        min_area_points=5,
    )
    
    # Filter by region
    bursts_filtered = filter_bursts_by_region(
        bursts,
        time_range=(0.95, 1.25),
        freq_range=(0, 50e3),
    )
    
    print(f"\nDetected {len(bursts_filtered)} bursts:")
    print_burst_summary(bursts_filtered)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sawtooth precursor analysis examples"
    )
    parser.add_argument(
        "--example",
        choices=["single", "multiple", "burst_detection"],
        default="single",
        help="Which example to run",
    )
    args = parser.parse_args()
    
    if args.example == "single":
        example_single_shot()
    elif args.example == "multiple":
        example_multiple_shots()
    elif args.example == "burst_detection":
        example_custom_burst_detection()
        
    print("All examples completed.")
