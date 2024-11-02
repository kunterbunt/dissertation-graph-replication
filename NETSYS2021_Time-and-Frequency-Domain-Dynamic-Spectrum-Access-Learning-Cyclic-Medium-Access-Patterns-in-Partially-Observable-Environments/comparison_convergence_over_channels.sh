#!/bin/bash -l
#SBATCH --job-name=comparison_convergence_over_channels
#SBATCH -o ./slurm_output/%j.out
#SBATCH -p smp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu 10G
#SBATCH --time=4-00:00:00
#SBATCH --constraint=OS8
#SBATCH --mail-type ALL
#SBATCH --mail-user sebastian.lindner@tuhh.de
make comparison_convergence_over_channels