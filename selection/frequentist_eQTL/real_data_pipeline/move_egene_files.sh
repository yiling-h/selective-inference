#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in $(seq -f %03g 1 100)
do
    #IN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/randomized_egene_names/
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/randomized_egene_names/
    #XIN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk${i}_mtx/
    XIN=/scratch/PI/sabatti/controlled_access_data/fastqtl_tmp/Liver/Liver_97_chunk${i}_mtx/
    #YIN=/scratch/PI/sabatti/controlled_access_data/fastqtl_sim/Liver_mixture7amp0.30/Liver_97_chunk${i}_mtx/
    YIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/Liver_97_chunk${i}_mtx/
    #SIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/Liver_97_chunk${i}_mtx/bon_output/
    SIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_snig/Liver_97_chunk${i}_mtx/bon_output/
    #OUT=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_simulated/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_snig/
    #PIN=/scratch/PI/jtaylo/snigdha_data/gtex/Liver_simulated/Liver_97_chunk${i}_mtx/
	sbatch move_egene_files.sbatch ${IN} ${XIN} ${YIN} ${SIN} $i ${OUT} 
done