#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in $(seq -f %03g 1 100)
do
    #IN=/scratch/PI/sabatti/controlled_access_data/temp_transfer/Muscle_Skeletal_mixture4amp0.30/Muscle_Skeletal_chunk${i}_mtx/
    #OUT=/scratch/PI/jtaylo/snigdha_data/gtex/simulation_muscle/Muscle_Skeletal_chunk${i}_mtx/
    #IN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk${i}_mtx/
    #OUT=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/Liver_97_chunk${i}_mtx/
    IN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk${i}_mtx/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/Liver_97_chunk${i}_mtx/
    mkdir -p ${OUT}
    #echo "${IN}"
    #echo "${OUT}"
	# bash single_simes_python.sbatch ${IN} ${OUT}
	sbatch simulated_single_simes_python.sbatch ${IN} ${OUT} $i
done