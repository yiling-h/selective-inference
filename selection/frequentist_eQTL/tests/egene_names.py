from __future__ import print_function
import numpy as np
import os
import sys

path = '/Users/snigdhapanigrahi/sim_nonrandomized_bon_output_liver/names'
inpath = '/Users/snigdhapanigrahi/sim_nonrandomized_bon_output_liver/nonrandomized_egenes/'
outpath = '/Users/snigdhapanigrahi/sim_nonrandomized_bon_output_liver/nonrandomized_egene_names/'

for i in range(100):

    gene_file = os.path.join(path + str(format(i+1, '03')) + "/Genes.txt")

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    sys.stderr.write("length" + str(len(content)) + "\n")

    index_file = np.loadtxt(os.path.join(inpath + "egene_index_" + str(format(i+1, '03')) + ".txt")).astype(int)

    if index_file.size>1:
        if index_file[0] != -1:
            egenes = [content[index_file[k]] for k in range(index_file.shape[0])]
            outfile = os.path.join(outpath + "egenes_" + str(format(i + 1, '03')) + ".txt")

            with open(outfile, 'w') as fo:
                for x in egenes:
                    fo.write(str(x) + '\n')

    elif index_file.size==1:
        if index_file != -1:
            egenes = content[index_file]
            outfile = os.path.join(outpath + "egenes_" + str(format(i + 1, '03')) + ".txt")

            with open(outfile, 'w') as fo:
                for x in egenes:
                    fo.write(str(x) + '\n')

    sys.stderr.write("iteration completed" + str(i+1) + "\n")



