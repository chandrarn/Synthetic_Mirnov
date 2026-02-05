#!/bin/bash -l

#SBATCH --job-name=SynthMirnov       # Name of the job
#SBATCH --output=result_%j.out # Output file (%j expands to jobId)
#SBATCH --error=result_%j.err  # Error file
#SBATCH --partition=sched_mit_psfc_r8     # Partition/queue to run in
#SBATCH --nodes=1              # Number of nodes
##SBATCH --ntasks=1             # Total tasks
#SBATCH --cpus-per-task=21      # CPU cores per task
##SBATCH --mem=8G               # Total memory (e.g., 8 GB)
#SBATCH --time=08:00:00        # Time limit hrs:min:sec
##SBATCH --mail-type=END        # Email alert at end
##SBATCH --mail-user=you@example.com # Email address

# Load necessary modules
source /home/rianc/venv/bin/activate

# Create a unique temporary folder to store working files in to avoid concurrent read conflicts
SCRIPT_DIR="/home/rianc/Documents/Synthetic_Mirnov/signal_generation/"
SLURM_WORKING_FOLDER="${SCRIPT_DIR}batch_runs/slurm_job_${SLURM_JOB_ID}/"
mkdir -p $SLURM_WORKING_FOLDER'input_data/'
INPUT_MESH="${SCRIPT_DIR}input_data/C_Mod_ThinCurr_Combined-homology.h5"
UNIQUE_MESH="${SCRIPT_DIR}C_Mod_ThinCurr_Combined-homology_job.h5"


echo $SCRIPT_DIR
echo $INPUT_MESH
echo $UNIQUE_MESH

# Copy the mesh file with unique name
if [ -f "$INPUT_MESH" ]; then
    cp "$INPUT_MESH" "$SLURM_WORKING_FOLDER""input_data/"
    echo "Copied mesh file to: $SLURM_WORKING_FOLDER""input_data/"
    export SLURM_WORKING_FOLDER
    export SCRIPT_DIR
else
    echo "Error: Input mesh file not found at $INPUT_MESH"
    exit 1
fi

# Run the command
export PYTHONPATH=$PYTHONPATH:"/home/rianc/SynthWave/"
cd $SLURM_WORKING_FOLDER

echo "Running synthetic spectrogram generation..."

srun python /home/rianc/Documents/Synthetic_Mirnov/signal_generation/batch_run_synthetic_spectrogram.py

# Cleanup: Remove the unique job folder with mesh/etc after the job completes
cd '../'
if [ -f "$UNIQUE_MESH" ]; then
    rm -rf "$SLURM_WORKING_FOLDER"
    echo "Deleted temporary working folder: $SLURM_WORKING_FOLDER"
fi