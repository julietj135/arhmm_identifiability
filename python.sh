#!/bin/bash
#SBATCH --job-name=test_gpu             # Job name
#SBATCH --mem=60000                     # Job memory request
#SBATCH -t 0-00:10:00               # Time limit days-hrs:min:sec
#SBATCH -N 1                         # requested number of nodes (usually just 1)
#SBATCH -p scavenger-gpu             # requested partition on which the job will run
#SBATCH --gres=gpu:1 
#SBATCH --exclusive
#SBATCH --output=outputs/test_gpu.out   # file path to slurm output
# SBATCH --array=0-9                 # job array

# python work.py $SLURM_ARRAY_TASK_ID
python test.py

