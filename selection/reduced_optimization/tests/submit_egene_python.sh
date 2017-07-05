#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed

OUT=/scratch/users/snigdha/freq_cis_eqtl/

mkdir -p $DIR

IN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk001_mtx/

for i in {0..50}
do
	sbatch single_egene_python.sbatch $IN $OUT $i
done