#!/bin/bash
#SBATCH --job-name="posi-og"
#SBATCH --ntasks=750
#SBATCH --mem-per-cpu=6gb
#SBATCH --time=8:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
# The application(s) to execute along with its input arguments and options:

python -m scoop posterior-run-og.py
