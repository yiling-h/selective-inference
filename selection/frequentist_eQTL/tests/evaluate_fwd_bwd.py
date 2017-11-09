import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/fwd_bwd_inference_10/inference'

df_master = pd.DataFrame()

allFiles = glob.glob(inf_path + "/*.txt")
columns = ["lower_ci", "upper_ci", "point_estimator", "length", "gene_name", "method", "nsignificant", "norm"]
i = 0
check_sig = 0.
check_norm = 0.
check_length = 0.

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
        data_naive = data[:, np.array([1, 2, 4, 3])]
        nsignals = int(data[0,0])
        nsig = int(data[0,5])
        norm = (np.power(data[:, 4], 2).sum())/float(nactive)
        if data[:,3].sum()/float(nactive)<10.:
            check_norm += norm
            check_length += data[:, 3].sum() / float(nactive)
            df_naive = pd.DataFrame(data=data_naive, columns=["lower_ci", "upper_ci", "point_estimator", "length"])
            df_naive = df_naive.assign(gene_name=gene,
                                       method="fwdbwd",
                                       nsignificant=nsig,
                                       norm=norm)

            check_sig += nsig / float(nactive)

        i = i + 1

    elif nactive == 1. and data[0] > 0.05:
        data_naive = data[np.array([1, 2, 4, 3])]
        nsignals = int(data[0])
        nsig = int(data[5])
        norm = np.power(data[4], 2)
        if data[3]<10.:
            check_norm += norm
            check_length += data[3]

            df_naive = pd.DataFrame(data=data_naive.reshape((1, 4)),
                                    columns=["lower_ci", "upper_ci", "point_estimator", "length"])
            df_naive['gene_name'] = gene
            df_naive['method'] = "fwdbwd"
            df_naive['nsignificant'] = nsig
            df_naive['norm'] = norm
            check_sig += nsig
        i = i + 1

    df_master = df_master.append(df_naive, ignore_index=True)

print("check significant", check_sig/1273., check_norm/1273., check_length/1273.)

print("count of total files", i)
df_master.to_csv("/Users/snigdhapanigrahi/inference_liver/real_fwd_bwd_inference_0.10.csv", index=False)
print("saved to file!")

