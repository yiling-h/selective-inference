#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in $(seq -f %03g 1 100)
do
    IN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk${i}_mtx/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/Liver_97_chunk${i}_mtx/
    mkdir -p ${OUT}
    DIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/Liver_97_chunk001_mtx/
	sbatch simulate_Y_chunk.sbatch ${IN} ${OUT} ${DIN}
done