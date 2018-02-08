import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/sim_inference_liver/additional_info_fwdbwd'
df_master = pd.DataFrame()

allFiles = glob.glob(inf_path + "/*.txt")
columns = ["coverage", "risk", "length", "gene_name", "num_true_sigs", "method", "fdr_screening",
           "fdr", "power", "full_risk"]
i = 0

check_coverage = 0.
check_risk = 0.
check_length = 0.
check_active = 0.

check_power_break = np.zeros(10)
negenes = np.zeros(10)

check_risk_full = 0.
check_power = 0.
check_fdr = 0.

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
        nsignals = int(data[0,0])
        data_naive = data[:, np.array([2, 4, 3])]
        check_coverage += data[:,2].sum()/float(nactive)
        check_risk += data[:,4].sum()/float(nactive)
        check_length += data[:,3].sum()/float(nactive)
        fdr_screening = data[0, 6]
        df_naive = pd.DataFrame(data=data_naive, columns=['coverage', 'risk', 'length'])
        df_naive = df_naive.assign(gene_name=gene,
                                   num_true_sigs=nsignals,
                                   method="fwdbwd",
                                   fdr_screening=fdr_screening,
                                   fdr=data[0, 6],
                                   inferential_power=data[0, 7],
                                   full_risk=data[0, 5])
        i = i + 1
        negenes[nsignals] += 1
        check_active += nactive

        check_risk_full += data[0, 5]
        check_power += data[0, 7]
        check_fdr += data[0, 6]

        check_power_break[nsignals] += data[0, 7]
        negenes[nsignals] += 1.

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
        df_naive['fdr'] = data[6]
        df_naive['fdr_screening'] = data[6]
        df_naive['inferential_power'] = data[7]
        df_naive['full_risk'] = data[5]
        i = i + 1

        negenes[nsignals] += 1
        check_active += nactive

        check_risk_full += data[5]
        check_power += data[7]
        check_fdr += data[6]

        check_power_break[nsignals] += data[7]
        negenes[nsignals] += 1.

    df_master = df_master.append(df_naive, ignore_index=True)

print("count of total files", i)
print("check significant", check_coverage/1535., check_risk/1535., check_length/1535., negenes, check_active/float(i))
df_master.to_csv("/Users/snigdhapanigrahi/sim_inference_liver/additional_fwd_bwd_inference_new.csv", index=False)
print("saved to file!")
print("power randomized versus nonrandomized", i, check_risk_full/float(i), check_power/float(i), check_fdr/float(i))
print("power break-up", np.true_divide(check_power_break,negenes))