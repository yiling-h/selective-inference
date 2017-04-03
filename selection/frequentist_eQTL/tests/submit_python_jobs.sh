#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed

DIR=/scratch/users/snigdha/freq_cis_eqtl/outputs/sparsity_3/level_1

mkdir -p $DIR

for i in {0..100}
do
	# bash single_python_run.sbatch $i $DIR
	sbatch single_python_run.sbatch $i $DIR
done