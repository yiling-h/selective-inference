#!/bin/bash

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate
cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/frequentist_eQTL/tests

DIR=/share/PI/sabatti/selective_inference/simulation

# P=7000
# M=5

MAX_SEED=100

for P in 5000 7000
do
    for M in 0 1 3 5 10
    do
        D_DIR=${DIR}/p${P}_s${M}
        mkdir -p ${DIR}/p${P}_s${M}
        python randomized_lasso.py gendata -p ${P} -m ${M} -s ${MAX_SEED} -o ${D_DIR}
    done
done






