#!/bin/bash
#SBATCH --job-name="posi-og"
#SBATCH --partition=compute
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=24
#SBATCH -t 04:00:00
#SBATCH -A TG-DMS190038

python -m scoop posterior-run-og.py
