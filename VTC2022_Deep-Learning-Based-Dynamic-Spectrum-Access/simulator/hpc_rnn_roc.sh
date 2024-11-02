#!/bin/bash
#SBATCH --job-name=i3_rnn_roc
#SBATCH --partition=smp
#SBATCH --constraint=OS8
### Configure resources
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gpus-per-task=0
#SBATCH --mem-per-cpu 3072
### Configure timeout
#SBATCH --time=2-00:00:00
### Configure emails
#SBATCH --mail-type ALL
#SBATCH --mail-user daniel.stolpmann@tuhh.de

source /fibus/fs2/1a/cei6676/i3_pythonenv/bin/activate

python3 main_rnn_roc.py noplot
