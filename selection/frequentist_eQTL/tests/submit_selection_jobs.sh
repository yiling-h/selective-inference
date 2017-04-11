#!/bin/bash

DIR=/share/PI/sabatti/selective_inference/simulation
# source /home/jjzhu/source_code/cis_eqtl_pipeline/.env/bin/activate
# cd /home/jjzhu/source_code/cis_eqtl_pipeline/selective-inference/selection/frequentist_eQTL/tests

MAX_SEED=100

for P in 5000 7000
do
    for M in 0 1 3 5 10
    do
        # for L in 0.8 1.0 1.2 1.4
        # for L in 0.6 1.8 
        for L in 0.6 
        do
            D_DIR=${DIR}/p${P}_s${M}/data
            R_DIR=${DIR}/p${P}_s${M}/res_rlasso${L}
            mkdir -p ${R_DIR}

            echo "" 
            echo "Processing ${D_DIR}" 
            # python randomized_lasso.py onlyselect -s ${MAX_SEED} -d ${D_DIR} -o ${R_DIR} -l ${L}
            sbatch single_python_run_selection.sbatch ${MAX_SEED} ${D_DIR} ${R_DIR} ${L}
        done
    done
done
