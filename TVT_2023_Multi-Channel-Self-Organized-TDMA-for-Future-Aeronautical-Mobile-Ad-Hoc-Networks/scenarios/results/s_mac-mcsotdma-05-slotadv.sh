#!/bin/bash -l
#SBATCH -p smp
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 6
#SBATCH --mem-per-cpu 16G
#SBATCH --time 3-00:00:00
#SBATCH --constraint=OS8

# Execute simulation
make sh-mac-mcsotdma-05 NUM_CPUS=6

# Exit job
exit
