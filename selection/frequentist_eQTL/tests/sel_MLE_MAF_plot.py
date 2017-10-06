import os, numpy as np
path = '/Users/snigdhapanigrahi/inference_liver/'
infile = os.path.join(path, "egenes_completed.txt")

inf_path = '/Users/snigdhapanigrahi/inference_liver/inference0/'
sel_path = '/Users/snigdhapanigrahi/inference_liver/sel_SNPs_with_MAFs/'

with open(infile) as g:
    eGenes = g.readlines()
eGenes = [x.strip() for x in eGenes]

MAF_snps = []
sel_MLE_snps = []
MLE_snps = []
pivots_snps = []
for i in range(len(eGenes)):

    if not os.path.exists(os.path.join(inf_path, "inference_" + str(eGenes[i]) + ".txt")) or \
            not os.path.exists(os.path.join(sel_path, "sel_SNPs_MAFs_" + str(eGenes[i]) + ".txt")):
            #or not os.path.exists(os.path.join(sel_path, "sel_SNPs_with_MAFs_" + str(eGenes[i]) + ".txt")):
        print("invalid iteration", i)
    else:

        results = np.loadtxt(os.path.join(inf_path, "inference_" + str(eGenes[i]) + ".txt"))
        MAF_infile = os.path.join(sel_path, "sel_SNPs_MAFs_" + str(eGenes[i]) + ".txt")
        with open(MAF_infile) as h:
            MAFs = h.readlines()

        if results.ndim > 1:
            nactive = results.shape[0]
        elif results.ndim==1 and results.shape[0]>0:
            nactive = 1.
        elif results.ndim==1 and results.shape[0]==0:
            nactive = 0.

        if nactive > 1 and results[:, 8].sum() / float(nactive) < 5:
            if results[:,4].shape[0] != results[:, 5].shape[0] or results[:,4].shape[0]!= np.asarray([float(x.strip().split('\t')[2]) for x in MAFs]).shape[0]:
                print("bad")
            sel_MLE_snps.append(results[:,4])
            MLE_snps.append(results[:, 5])
            pivots_snps.append(results[:, 6])
            MAF_snps.append(np.asarray([float(x.strip().split('\t')[2]) for x in MAFs]))
        elif nactive == 1. and results[8] / float(nactive) < 5:
            sel_MLE_snps.append(np.array([results[4]]))
            MLE_snps.append(np.array([results[5]]))
            pivots_snps.append(np.array(results[6]))
            MAF_snps.append(np.asarray([float(x.strip().split('\t')[2]) for x in MAFs]))

MAF_snps = np.hstack(MAF_snps)
sel_MLE_snps = np.hstack(sel_MLE_snps)
MLE_snps = np.hstack(MLE_snps)
pivots_snps = np.hstack(pivots_snps)

print("shapes", MAF_snps.shape, sel_MLE_snps.shape, MLE_snps.shape, pivots_snps.shape)
inf_snps = np.transpose(np.vstack((MAF_snps, pivots_snps, sel_MLE_snps, MLE_snps)))
np.savetxt("/Users/snigdhapanigrahi/inference_liver/pivots_mle_MAFs.txt", inf_snps)
