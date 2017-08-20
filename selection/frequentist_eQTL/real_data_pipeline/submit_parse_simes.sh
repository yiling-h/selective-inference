#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in $(seq -f %03g 1 100)
do
    #IN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/randomized_hs_egene_names/
    #OUT=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/Liver_97_chunk${i}_mtx/bon_hs_output/
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/randomized_egene_names/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/Liver_97_chunk${i}_mtx/bon_output/
    mkdir -p ${OUT}
    DIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/randomized_egene_boninfo/
	sbatch single_parse_simes.sbatch ${IN} ${OUT} $i ${DIN}
done