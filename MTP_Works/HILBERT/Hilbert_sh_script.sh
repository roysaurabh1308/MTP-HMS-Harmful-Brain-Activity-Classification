#!/bin/bash
#SBATCH --job-name=HIL_JOB5          # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=45gb                      # Job memory request
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --gpus-per-node=1               # Number of GPU
#SBATCH --partition=gpupart_24hour              # Time limit hrs:min:sec
#SBATCH --time=23:15:00             # Time limit hrs:min:sec
#SBATCH --output=hilbert_training_6.log        # Standard output and error log

conda activate mtp_env_v1

# Run the Python script with arguments
python3 hilbert_hms_v3_ce.py