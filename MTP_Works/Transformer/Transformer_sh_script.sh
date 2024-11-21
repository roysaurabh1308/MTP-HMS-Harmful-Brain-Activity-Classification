#!/bin/bash
#SBATCH --job-name=TRF_JOB0          # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=45gb                      # Job memory request
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --gpus-per-node=1               # Number of GPU
#SBATCH --partition=gpupart_p100              # Time limit hrs:min:sec
#SBATCH --time=1-23:15:00             # Time limit hrs:min:sec
#SBATCH --output=transformer_training_0.log        # Standard output and error log

conda activate mtp_env_v1

# Run the Python script with arguments
python3 Trans_hms_v1_ce_tf.py