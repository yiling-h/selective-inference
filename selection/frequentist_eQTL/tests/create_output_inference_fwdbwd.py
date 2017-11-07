import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/sim_fwd_bwd_inference/inference'
#inf_path =r'/Users/snigdhapanigrahi/data_split_inference/inference'
df_master = pd.DataFrame()

allFiles = glob.glob(inf_path + "/*.txt")
#columns = ["lower_ci", "upper_ci", "point_estimator", "length", "gene_name", "method"]
columns = ["coverage", "risk", "length", "gene_name", "num_true_sigs", "method"]
i = 0

check_coverage = 0.
check_risk = 0.
check_length = 0.
negenes = np.zeros(10)
for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[3]
    print("gene", gene)


    data = np.loadtxt(file_)

    if data.ndim > 1:
        nactive = data.shape[0]
    elif data.ndim == 1 and data.shape[0] > 0:
        nactive = 1.
    elif data.ndim == 1 and data.shape[0] == 0:
        nactive = 0.

    if nactive > 1:
        data_naive = data[:, np.array([2, 4, 3])]
        nsignals = int(data[0,0])
        check_coverage += data[:,2].sum()/float(nactive)
        check_risk += data[:,4].sum()/float(nactive)
        check_length += data[:,3].sum()/float(nactive)
        df_naive = pd.DataFrame(data=data_naive, columns=['coverage', 'risk', 'length'])
        df_naive = df_naive.assign(gene_name=gene,
                                   num_true_sigs=nsignals,
                                   method="fwdbwd")
        i = i + 1
        negenes[nsignals] += 1

    elif nactive == 1. and data[1] > 0.05:
        data_naive = data[np.array([2, 4, 3])]
        nsignals = int(data[0])
        check_coverage += data[2]
        check_risk += data[4]
        check_length += data[3]
        df_naive = pd.DataFrame(data=data_naive.reshape((1, 3)), columns=['coverage', 'risk', 'length'])
        df_naive['gene_name'] = gene
        df_naive['num_true_sigs'] = nsignals
        df_naive['method'] = "fwdbwd"
        i = i + 1

        negenes[nsignals] += 1

    df_master = df_master.append(df_naive, ignore_index=True)

print("count of total files", i)
print("check significant", check_coverage/505., check_risk/505., check_length/505., negenes)
#df_master.to_csv("/Users/snigdhapanigrahi/sim_inference_liver/pruned_fwd_bwd_inference.csv", index=False)
print("saved to file!")



