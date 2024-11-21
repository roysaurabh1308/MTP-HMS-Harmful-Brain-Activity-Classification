#!/bin/bash
#SBATCH --job-name=ViT_JOB0          # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=30gb                      # Job memory request
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --gpus-per-node=1               # Number of GPU
#SBATCH --partition=gpupart_1hour              # Time limit hrs:min:sec
#SBATCH --time=00:59:00             # Time limit hrs:min:sec
#SBATCH --output=vit_training.log        # Standard output and error log

conda activate mtp_hms10_env

# Run the Python script with arguments
python3 ViT_test.py