from __future__ import print_function
import numpy as np
import os
import sys

if __name__ == "__main__":

    path = sys.argv[1]
    result = sys.argv[5]
    infile = os.path.join(path, "egenes_" + str(result) + ".txt")
    with open(infile) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    sys.stderr.write("length" + str(len(content)) + "\n")

    outdir = sys.argv[6]
    X_path = sys.argv[2]
    Y_path = sys.argv[3]
    bon_path = sys.argv[4]

    for i in range(len(content)):

        X = np.load(os.path.join(X_path + "X_" + str(content[i])) + ".npy")
        out_X = os.path.join(outdir + "X_" + str(content[i]) + ".npy")
        np.save(out_X, X)

        y = np.load(os.path.join(X_path + "y_" + str(content[i])) + ".npy")
        y = y.reshape((y.shape[0],))
        out_Y = os.path.join(outdir + "y_" + str(content[i]) + ".npy")
        np.save(out_Y, y)

        prototypes = np.loadtxt(os.path.join(Y_path + "protoclust_"  + str(content[i])) + ".txt", delimiter='\t')
        outfile = os.path.join(outdir + "protoclust_" + str(content[i]) + ".txt")
        np.savetxt(outfile, prototypes, delimiter='\t')

        simes_output = np.loadtxt(os.path.join(bon_path + "simes_"  + str(content[i])) + ".txt")
        simesfile = os.path.join(outdir + "simes_" + str(content[i]) + ".txt")
        np.savetxt(simesfile, simes_output)

        sys.stderr.write("egene completed" + str(i) + "\n")





