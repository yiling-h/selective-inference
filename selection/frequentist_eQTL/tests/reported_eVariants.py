import numpy as np
import glob, os
import pandas as pd

sel_path = r'/Users/snigdhapanigrahi/inference_liver/sel_SNPs_with_MAFs/'
#inf_path = r'/Users/snigdhapanigrahi/inference_liver/inference0'
inf_path = r'/Users/snigdhapanigrahi/inference_liver/inference_05/'
#gene_path = r'/Users/snigdhapanigrahi/inference_liver/common_egenes.txt'
gene_path = r'/Users/snigdhapanigrahi/inference_liver/eGenes_05.txt'
outpath = r'/Users/snigdhapanigrahi/inference_liver/eVariants_05/'

#print(os.path.join(outpath, "eVariants/"))
with open(gene_path) as g:
    eGenes = g.readlines()
eGenes = [x.strip() for x in eGenes]

for i in range(len(eGenes)):
    #if os.path.exists(os.path.join(inf_path, "inference_" + str(eGenes[i]) + ".txt")) and \
    #        os.path.exists(os.path.join(sel_path, "sel_SNPs_MAFs_" + str(eGenes[i]) + ".txt")):
    if os.path.exists(os.path.join(sel_path, "sel_SNPs_MAFs_" + str(eGenes[i]) + ".txt")):
        print("gene name", str(eGenes[i]))
        MAF_infile = pd.read_csv(os.path.join(sel_path, "sel_SNPs_MAFs_" + str(eGenes[i]) + ".txt"), sep="\t",
                                 header=None)
        MAF_infile.columns = ["SNP_index", "eVariant", "MAF"]
        index = np.asarray(MAF_infile['SNP_index'])

        data = np.loadtxt(os.path.join(inf_path, "inference_" + str(eGenes[i]) + ".txt"))
        if data.ndim > 1:
            nactive = data.shape[0]
        elif data.ndim == 1 and data.shape[0] > 0:
            nactive = 1.
        elif data.ndim == 1 and data.shape[0] == 0:
            nactive = 0.

        index_sel = []
        if nactive > 1:
            for k in range(int(nactive)):
                if data[k,1]<0. or data[k,0]>0.:
                    index_sel.append(int(data[k,9]))

        elif nactive == 1:
            if data[1] < 0. or data[0] > 0.:
                index_sel.append(int(data[9]))

        print("shape", index_sel)
        outfile = os.path.join(outpath + "e_"+ str(eGenes[i]) + ".txt")
        np.savetxt(outfile, np.asarray(index_sel))

# E = np.load('/Users/snigdhapanigrahi/inference_liver/s_ENSG00000187642.5.npy').astype(int)
# if E.ndim == 0:
#     E = np.asarray([E])
# print("E", E, E.shape)
# E = E.reshape((E.shape[0],))
