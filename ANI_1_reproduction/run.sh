#!/bin/bash

#SBATCH -A m3718
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -t 04:00:00
#SBATCH -c 8
#SBATCH --job-name=run_ani_training
#SBATCH -q regular
#SBATCH --wrap=hostname
#SBATCH --mail-user=ronyadin@berkeley.edu
#SBATCH --mail-type=ALL

module load cuda
source activate ani_1

srun python3 run_ani.py -d /global/cfs/cdirs/m3718/data/ANI-1_release/ani_gdb_s01.h5,/global/cfs/cdirs/m3718/data/ANI-1_release/ani_gdb_s02.h5 -c ./0516_checkpoint_defaults.pt -e 15 -b 256 -r 42 --use-gpu
