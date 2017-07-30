#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in $(seq -f %03g 1 100)
do
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/real_Liver/randomized_egene_names
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/real_Liver/Liver_97_chunk${i}_mtx/prototypes
    mkdir -p ${OUT}
    DIN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk${i}_mtx
	sbatch single_protoclust.sbatch ${IN} ${OUT} $i ${DIN}
done