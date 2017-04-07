#!/bin/bash 
#SBATCH --job-name=job
#SBATCH --output=jobs/%j.out
#SBATCH --error=jobs/%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=4000
#SBATCH -p hns,normal 

MAX_SEED=$1
D_DIR=$2
R_DIR=$3
L=$4

# cd to program directory
source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate
cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/frequentist_eQTL/tests

CMD="python randomized_lasso.py onlyselect -s ${MAX_SEED} -d ${D_DIR} -o ${R_DIR} -l ${L}"

echo $CMD

$CMD
