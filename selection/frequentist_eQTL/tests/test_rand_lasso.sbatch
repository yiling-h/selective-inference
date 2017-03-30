#!/bin/bash 
#SBATCH --job-name=job
#SBATCH --output=jobs/%j.out
#SBATCH --error=jobs/%j.err
#SBATCH --time=00:00:10
#SBATCH -p hns
#SBATCH -n 4
#SBATCH -N 1 

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate

cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/frequentist_eQTL/tests

echo "Number of processers available:"
nproc

echo "Running python script"
python randomized_lasso.py test -n4
