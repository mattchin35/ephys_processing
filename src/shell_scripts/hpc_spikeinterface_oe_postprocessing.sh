#!/bin/bash

#SBATCH --partition=normal
#SBATCH --job-name=spikeinterface_oe_postprocess
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/gs/gsfs0/users/mchin1/logs/spikeinterface_oe_postprocess_%j.log

set -euo pipefail

cd "/gs/gsfs0/users/mchin1/ephys_processing"
uv run --frozen python src/postprocessing/spikeinterface_oe_postprocessing.py
