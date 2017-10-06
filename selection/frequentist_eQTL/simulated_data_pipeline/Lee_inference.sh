#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


#for i in 17
do
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_simulated/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_simulated/Lee_inf/
	sbatch submit_Lee.sbatch ${IN} ${OUT} $i
done