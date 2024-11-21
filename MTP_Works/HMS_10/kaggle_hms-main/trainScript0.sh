#!/bin/bash
#SBATCH --job-name=ViT_JOB0          # Job name
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --mem=30gb                      # Job memory request
#SBATCH --cpus-per-task=2                # Number of CPU cores per task
#SBATCH --gpus-per-node=2               # Number of GPU
#SBATCH --partition=gpupart_48hour              # Time limit hrs:min:sec
#SBATCH --time=1-23:50:45             # Time limit hrs:min:sec
#SBATCH --output=train_logs.log        # Standard output and error log

conda activate mtp_hms10_env

# Run the Python script with arguments
export PYTHONWARNINGS="ignore"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=localhost:29400 code/run_train.py --exp exp_stage1_6