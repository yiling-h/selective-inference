from __future__ import print_function
import numpy as np
import os
import sys

if __name__ == "__main__":

    path = sys.argv[1]
    outdir = sys.argv[2]
    result = sys.argv[3]
    dinpath = sys.argv[4]

    infile = os.path.join(path, "egenes_" + str(result) + ".txt")

    dinfile = np.loadtxt(os.path.join(dinpath, "egene_index_" + str(result) + ".txt"))

    with open(infile) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    sys.stderr.write("length" + str(len(content)) + "\n")

    for i in range(len(content)):

        outfile = os.path.join(outdir, "simes_" + str(content[i]) + ".txt")
        np.savetxt(outfile, dinfile[i,:])
        sys.stderr.write("egene completed" + str(i) + "\n")
