#!/bin/bash

DIR=/share/PI/sabatti/selective_inference/simulation

MAX_SEED=100

for P in 5000 7000
do
    for M in 0 1 3 5 10
    do
        for L in 0.8 1.0 1.2 1.4
        do
            D_DIR=${DIR}/p${P}_s${M}/data
            R_DIR=${DIR}/p${P}_s${M}/res_rlasso${L}
            mkdir -p ${R_DIR}

            echo "" 
            echo "Processing ${D_DIR}" 
            # python randomized_lasso.py onlyselect -s ${MAX_SEED} -d ${D_DIR} -o ${R_DIR} -l ${L}
            sbatch single_python_run_selection.sh ${MAX_SEED} ${D_DIR} ${R_DIR} ${L}
        done
    done
done
