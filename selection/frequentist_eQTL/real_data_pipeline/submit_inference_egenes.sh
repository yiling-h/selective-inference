#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in {0..499}
do
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_snig/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_snig/inference/
	sbatch inference_files.sbatch ${IN} ${OUT} $i
done