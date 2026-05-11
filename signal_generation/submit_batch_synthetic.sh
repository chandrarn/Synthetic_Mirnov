#!/bin/bash
#SBATCH --job-name=synth_mirnov
#SBATCH --partition=sched_mit_psfc_r8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --mem-per-cpu=4G
#SBATCH --time=8:00:00
#SBATCH --output=slurm_logs/synth_mirnov_%j.out
#SBATCH --error=slurm_logs/synth_mirnov_%j.err

set -eo pipefail

# If not already running inside a SLURM job, re-submit via sbatch and exit
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Not running under SLURM — submitting via sbatch..."
    sbatch "$0"
    exit $?
fi

# Activate virtual environment
source ~/venv/bin/activate

# Force Python stdout/stderr to flush immediately so all output reaches the log file
export PYTHONUNBUFFERED=1

echo "Starting batch_run_synthetic_spectrogram at $(date)"
echo "Running on host: $(hostname)"
echo "CPUs available: ${SLURM_CPUS_PER_TASK:-unset}"

# Create log dir relative to submission directory (SLURM_SUBMIT_DIR is set by sbatch)
mkdir -p "${SLURM_SUBMIT_DIR}/slurm_logs"

# Run from signal_generation/ so relative paths (input_data/, training_data/, etc.) resolve correctly
cd "${SLURM_SUBMIT_DIR}/signal_generation"

# Python -u only flushes Python streams. OFT/ThinCurr native stdout can still block-buffer
# when SLURM redirects output to a file, so run the process under a PTY to force line buffering.
if command -v script >/dev/null 2>&1; then
    script -efq -c "python -u batch_run_synthetic_spectrogram.py" /dev/null
else
    stdbuf -oL -eL python -u batch_run_synthetic_spectrogram.py
fi

echo "Finished at $(date)"
