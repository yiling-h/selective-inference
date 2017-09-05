#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in {0..199}
do
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_simulated/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_simulated/inference0/
	sbatch inference_files.sbatch ${IN} ${OUT} $i
done