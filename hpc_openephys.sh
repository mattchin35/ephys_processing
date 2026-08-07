#!/bin/bash

# Which partition to use
#SBATCH -p normal

# Name to identify job
#SBATCH --job-name=openephys_preprocess

# Number of threads
#SBATCH -n 16

# Number of tasks total (1 per node)
#SBATCH --ntasks=1

# Number of tasks per node
#--tasks-per-node=1

# Memory per node.
#SBATCH --mem=128gb

# Time limit to run script
#SBATCH -t 24:00:00

# Where to save the output (log)
#SBATCH -o /gs/gsfs0/users/mchin1/logs/openephys_preprocess_%j.log

#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)

#SBATCH --mail-user=matthew.chin@einsteinmed.edu # Where to send mail

set -e

echo "Sourcing .bashrc..."
source ~/.bashrc

echo "Activating conda environment..."
conda activate spikeinterface

set -u

echo "Changing to project directory..."
cd "$HOME/ephys_processing"

export PYTHONPATH="$HOME/ephys_processing:$HOME/ephys_processing/src/preprocessing:${PYTHONPATH:-}"

echo "Running Open Ephys HPC workflow..."
python main_oe_hpc.py "$@"

echo "Done!"
exit 0
