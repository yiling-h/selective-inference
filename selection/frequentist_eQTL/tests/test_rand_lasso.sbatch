#!/bin/bash 
#SBATCH --job-name=job
#SBATCH --output=jobs/%j.out
#SBATCH --error=jobs/%j.err
#SBATCH --time=00:00:10
#SBATCH -p hns
#SBATCH -n 4

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate

cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/frequentist_eQTL/tests

python randomized_lasso.py test -n2 
