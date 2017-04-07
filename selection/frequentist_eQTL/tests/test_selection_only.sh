#!/bin/bash

source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate
cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/frequentist_eQTL/tests

DIR=/share/PI/sabatti/selective_inference/simulation

# MAX_SEED=100
MAX_SEED=2

# for P in 5000 7000
for P in 5000
do
    # for M in 0 1 3 5 10
    for M in 5
    do
        for L in 1.0 1.2
        do
            D_DIR=${DIR}/p${P}_s${M}/data
            R_DIR=${DIR}/p${P}_s${M}/res_rlasso${L}
            mkdir -p ${R_DIR}

            echo "" 
            echo "Processing ${D_DIR}" 
            python randomized_lasso.py onlyselect -s ${MAX_SEED} -d ${D_DIR} -o ${R_DIR} -l ${L}
        done
    done
done
