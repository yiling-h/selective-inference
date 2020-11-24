#!/bin/bash
#SBATCH --job-name="posi-het"
#SBATCH --partition=compute
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=24
#SBATCH -t 04:00:00
#SBATCH -A TG-DMS190038

python -m scoop posterior-run-hetero.py
