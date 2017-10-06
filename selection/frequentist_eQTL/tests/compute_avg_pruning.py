from __future__ import print_function
import sys
import os
import numpy as np

if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]
    p_prepruned = 0.
    p_pruned = 0.
    for egene in range(len(content)):
        gene = str(content[egene])
        X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
        n, p = X.shape
        p_prepruned += p

        prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
        prototypes = np.unique(prototypes).astype(int)
        p_pruned += prototypes.shape[0]
        
    sys.stderr.write("avg" + str(p_pruned/float(len(content))) + "\n")
    sys.stderr.write("avg" + str(p_prepruned/float(len(content))) + "\n")
