#!/bin/bash
#SBATCH --job-name="posi-het"
#SBATCH --ntasks=305
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=06-00:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
# The application(s) to execute along with its input arguments and options:

python -m scoop posterior-run-hetero.py
