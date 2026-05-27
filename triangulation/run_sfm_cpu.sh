#!/bin/bash
#SBATCH -A berzelius-2025-319 
#SBATCH -J sfm_hloc               
#SBATCH -t 00-12:00:00               
#SBATCH -o log_file/sfm_cpu_step_0to12%j.log

#SBATCH -p berzelius-cpu                    
#SBATCH --nodes=1                    
#SBATCH --cpus-per-task=64        
#SBATCH --mem=400G 
echo "Running in CPU mode on $HOSTNAME"

python triangulation_cpu_steps.py