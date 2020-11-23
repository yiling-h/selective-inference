#!/bin/bash
#SBATCH --job-name="oposi-het"
#SBATCH --partition=compute
#SBATCH --nodes=40
#SBATCH --ntasks-per-node=24
#SBATCH -t 01:00:00
#SBATCH -A TG-DMS190038

python -m scoop posterior-run-atomic.py
