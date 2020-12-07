#!/bin/bash
#SBATCH --job-name="posi-het"
#SBATCH --ntasks=75
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=8:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
# The application(s) to execute along with its input arguments and options:

python -m scoop posterior-run-hetero.py
