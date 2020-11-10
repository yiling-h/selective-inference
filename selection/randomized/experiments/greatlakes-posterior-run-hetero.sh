#!/bin/bash
#SBATCH --job-name="ogposi"
#SBATCH --ntasks=190
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=1:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
# The application(s) to execute along with its input arguments and options:

python -m scoop posterior-run-hetero.py
