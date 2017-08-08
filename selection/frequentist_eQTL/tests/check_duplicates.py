from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/'

    gene_1 = str("ENSG00000230092.3")
    X_1 = np.load(os.path.join(path + "X_" + gene_1) + ".npy")

    gene_2 = str("ENSG00000225880.4")
    X_2 = np.load(os.path.join(path + "X_" + gene_2) + ".npy")

    prototypes_1 = np.loadtxt(
        os.path.join("/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/protoclust_" + gene_1) + ".txt",
        delimiter='\t')

    prototypes_2 = np.loadtxt(
        os.path.join("/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/protoclust_" + gene_2) + ".txt",
        delimiter='\t')

    print("check close", np.unique(prototypes_1) - np.unique(prototypes_2))