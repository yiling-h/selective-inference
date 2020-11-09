#!/bin/bash
#SBATCH --job-name="ogposi"
#SBATCH --ntasks=40
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=20:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
# The application(s) to execute along with its input arguments and options:

python -m scoop posterior-run.py
