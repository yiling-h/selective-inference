import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/sim_inference_liver/inference0'
sel_path =r'/Users/snigdhapanigrahi/sim_inference_liver/nonrand_sel'
allFiles = glob.glob(inf_path + "/*.txt")

check_power_break = np.zeros(10)
check_fdr_break = np.zeros(10)
nsig_break = np.zeros(10)
nextreme = 0.
i = 0

for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    print("gene", gene)
    i= i+1
    sel = np.loadtxt(os.path.join(sel_path, "nonrand_sel_" + str(gene) + ".txt"))

    if df.ndim > 1:
        nactive = df.shape[0]
    elif df.ndim == 1 and df.shape[0] > 0:
        nactive = 1.
    elif df.ndim == 1 and df.shape[0] == 0:
        nactive = 0.
    nsignals = int(sel[6])

    if nactive > 1:
        nsig_break[nsignals] += 1.
        check_power_break[nsignals] += df[0,13]
        check_fdr_break[nsignals] += df[0,14]
    elif nactive == 1.:
        nsig_break[nsignals] += 1.
        check_power_break[nsignals] += df[13]
        check_fdr_break[nsignals] += df[14]
    else:
        print("iteration", i)

print("count of total files", i)
print("power break-up", nsig_break, check_power_break, check_fdr_break)
print("power break up", np.true_divide(check_power_break,nsig_break), np.true_divide(check_fdr_break,nsig_break))



