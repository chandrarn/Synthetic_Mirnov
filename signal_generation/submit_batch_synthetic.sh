#!/bin/bash
#SBATCH --job-name=synth_mirnov
#SBATCH --partition=sched_mit_psfc_r8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --output=../slurm_logs/synth_mirnov_%j.out
#SBATCH --error=../slurm_logs/synth_mirnov_%j.err

set -euo pipefail

# Activate virtual environment
source ~/venv/bin/activate

# Run from signal_generation/ so relative paths (input_data/, training_data/, etc.) resolve correctly
cd "$(dirname "$0")"

echo "Starting batch_run_synthetic_spectrogram at $(date)"
echo "Running on host: $(hostname)"
echo "CPUs available: $SLURM_CPUS_PER_TASK"

mkdir -p ../slurm_logs

python batch_run_synthetic_spectrogram.py

echo "Finished at $(date)"
