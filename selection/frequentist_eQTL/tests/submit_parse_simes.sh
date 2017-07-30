#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in $(seq -f %03g 1 100)
do
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/randomized_egene_names/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/Liver_97_chunk${i}_mtx/simes_output/
    mkdir -p ${OUT}
    DIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/randomized_egene_simesinfo/
	sbatch single_parse_simes.sbatch ${IN} ${OUT} $i ${DIN}
done