#!/bin/bash
# Setup bash job headers

# load local environment

# setup dir if needed


for i in {0..21}
do
    IN=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_snig/
    OUT=/scratch/PI/jtaylo/snigdha_data/gtex/egene_Liver_snig/nonrand_sel/
	sbatch overlap_nonrandomized.sbatch ${IN} ${OUT} $i
done