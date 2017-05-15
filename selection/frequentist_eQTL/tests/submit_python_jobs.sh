#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed

DIR=/scratch/users/snigdha/freq_cis_eqtl/outputs/high_dim_test_0

mkdir -p $DIR

for i in {0..100}
do
	# bash single_python_run.sbatch $i $DIR
	sbatch single_python_run.sbatch $i $DIR
done