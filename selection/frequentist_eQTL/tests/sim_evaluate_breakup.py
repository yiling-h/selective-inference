
import os, numpy as np

path = '/Users/snigdhapanigrahi/sim_inference_liver/'
infile = os.path.join(path, "eGenes.txt")

inf_path = '/Users/snigdhapanigrahi/sim_inference_liver/inference0/'
sel_path =r'/Users/snigdhapanigrahi/sim_inference_liver/nonrand_sel'

with open(infile) as g:
    eGenes = g.readlines()
eGenes = [x.strip() for x in eGenes]

coverage_ad = np.zeros(10)
coverage_unad = np.zeros(10)
risk_ad = np.zeros(10)
risk_unad = np.zeros(10)
length_ad = np.zeros(10)
length_unad = np.zeros(10)
power = np.zeros(10)
fdr = np.zeros(10)
negenes = np.zeros(10)

for i in range(len(eGenes)):

    if not os.path.exists(os.path.join(inf_path, "inference_" + str(eGenes[i]) + ".txt")):
        print("iteration", i)
    else:
        results = np.loadtxt(os.path.join(inf_path, "inference_" + str(eGenes[i]) + ".txt"))

        if results.ndim > 1:
            nactive = results.shape[0]
        elif results.ndim==1 and results.shape[0]>0:
            nactive = 1.
        elif results.ndim==1 and results.shape[0]==0:
            nactive = 0.

        sel = np.loadtxt(os.path.join(sel_path, "nonrand_sel_" + str(eGenes[i]) + ".txt"))

        nsignals = int(sel[6])

        if nactive > 1 and results[:, 8].sum() / float(nactive) < 15.:
            coverage_ad[nsignals] += results[:, 6].sum() / float(nactive)
            coverage_unad[nsignals] += results[:, 7].sum() / float(nactive)
            risk_ad[nsignals] += results[:, 8].sum() / float(nactive)
            risk_unad[nsignals] += results[:, 9].sum() / float(nactive)
            length_ad[nsignals] += results[:, 10].sum() / float(nactive)
            length_unad[nsignals] += results[:, 11].sum() / float(nactive)
            power[nsignals] += results[:, 13][0]
            fdr[nsignals] += results[:, 14][0]
            negenes[nsignals] += 1.

        elif nactive == 1. and results[8].sum() / float(nactive) < 15.:
            coverage_ad[nsignals] += results[6]
            coverage_unad[nsignals] += results[7]
            risk_ad[nsignals] += results[8]
            risk_unad[nsignals] += results[9]
            length_ad[nsignals] += results[10]
            length_unad[nsignals] += results[11]
            power[nsignals] += results[13]
            fdr[nsignals] += results[14]
            negenes[nsignals] += 1.


print("break-up", np.true_divide(coverage_ad, negenes), np.true_divide(coverage_unad, negenes),
          np.true_divide(risk_ad, negenes), np.true_divide(risk_unad, negenes),
          np.true_divide(length_ad, negenes),  np.true_divide(length_unad, negenes),
          np.true_divide(power, negenes), np.true_divide(fdr, negenes), negenes/1764.)






