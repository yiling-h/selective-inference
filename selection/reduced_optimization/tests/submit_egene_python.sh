#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed

OUT=/scratch/users/snigdha/freq_cis_eqtl/egene_0

mkdir -p $OUT

IN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk001_mtx/

for i in {0..100}
do
	sbatch submit_egene_python.sbatch $IN $OUT $i
done