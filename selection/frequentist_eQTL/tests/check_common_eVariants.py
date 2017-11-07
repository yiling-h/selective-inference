from __future__ import print_function
import sys
import os
import numpy as np
from scipy.stats.stats import pearsonr

if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "common_egenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    for egene in range(len(content)):
        gene = str(content[egene])
        if os.path.exists(os.path.join(inpath, "eVariants/e_" + gene + ".txt")):
            X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
            n, p = X.shape
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))

            S = (np.load(os.path.join(inpath + "s_" + gene) + ".npy")).astype(int)
            S = S.reshape((S.shape[0],))
            S = S - 1

            E = (np.loadtxt(os.path.join(inpath, "eVariants/e_" + str(content[egene]) + ".txt"))).astype(int)
            if E.ndim == 0:
                E = np.asarray([E])
            sys.stderr.write("Reported eVariants" + str(E) + "\n")
            E = E.reshape((E.shape[0],))

            corr = np.zeros((E.shape[0], S.shape[0]))

            for k in range(E.shape[0]):
                for j in range(S.shape[0]):
                    corr[k,j] = pearsonr(X[:, E[k]], X[:, S[j]])[0]

            outfile = os.path.join(outdir + "corr_" + gene + ".txt")
            np.savetxt(outfile, corr)