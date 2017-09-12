import os
import numpy as np

inpath =r'/Users/snigdhapanigrahi/inference_liver/inference0/'
path='/Users/snigdhapanigrahi/inference_liver/egenes_completed.txt'
protopath='/Users/snigdhapanigrahi/egene_liver/egene_Liver_snig/'
outdir='/Users/snigdhapanigrahi/inference_liver/sel_SNPs/'
with open(path) as g:
    cGenes = g.readlines()
cGenes = [x.strip() for x in cGenes]
count = 0
for i in range(len(cGenes)):

    inference = np.loadtxt(os.path.join(inpath + "inference_" + str(cGenes[i])) + ".txt")
    outfile = os.path.join(outdir, "sel_SNPs_" + str(cGenes[i]) + ".txt")
    if inference.size > 0:
        count+=1
        if inference.ndim > 1:
            nactive = inference.shape[0]
        else:
            nactive = 1

        prototypes = np.loadtxt(os.path.join(protopath + "protoclust_" + str(cGenes[i])) + ".txt", delimiter='\t')
        prototypes = np.unique(prototypes).astype(int)
        if nactive > 1:
            active_set = inference[:,7].astype(int)
            snp_indices = prototypes[active_set]
            np.savetxt(outfile, snp_indices)

        elif nactive == 1:
            active_set = inference[7].astype(int)
            active_set = active_set.reshape((1,))
            print("active set", active_set)
            snp_indices = prototypes[active_set]
            np.savetxt(outfile, snp_indices)

print("count", count)



