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

uv run --locked --project "$HOME/ephys_processing" "$HOME/ephys_processing/run_openephys_preprocessing.py" "$@"
