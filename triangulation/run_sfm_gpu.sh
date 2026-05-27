#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J sfm_hloc_gpu               
#SBATCH -t 00-12:00:00               
#SBATCH -o log_file/sfm_gpu_step_0to12%j.log

#SBATCH -p berzelius
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

echo "Running in GPU mode on $HOSTNAME"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

nvidia-smi

python triangulation_gpu_steps.py