import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/sim_inference_liver/additional_info_nonrand'

allFiles = glob.glob(inf_path + "/*.txt")
columns = ["coverage", "risk", "length", "gene_name", "num_true_sigs", "method", "fdr", "power", "full_risk"]
i = 0
df_master = pd.DataFrame()

check_fdr = 0.
check_power = 0.
check_risk_full = 0.

check_power_break = np.zeros(10)
nsig_break = np.zeros(10)

for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    print("gene", gene)

    i = i + 1
    data = np.loadtxt(file_)
    if data.ndim > 1:
        nactive = data.shape[0]
    elif data.ndim == 1 and data.shape[0] > 0:
        nactive = 1.
    elif data.ndim == 1 and data.shape[0] == 0:
        nactive = 0.

    if nactive > 1:
        nsignals = int(data[0, 9])
        data_selinf = data[:, np.array([0, 4, 2])]

        df_selinf = pd.DataFrame(data=data_selinf, columns=['coverage', 'risk', 'length'])
        df_selinf = df_selinf.assign(gene_name=str(gene),
                                     num_true_sigs=nsignals,
                                     method="Lee_nonrand",
                                     fdr=data[0,8],
                                     inferential_power=data[0,7],
                                     full_risk=data[0, 6])
        check_fdr += data[0,8]
        check_power += data[0,7]
        check_risk_full += data[0,6]

        check_power_break[nsignals] += data[0,7]
        nsig_break[nsignals] += 1.

    elif nactive == 1.:

        nsignals = int(data[9])
        data_selinf = data[np.array([0, 4, 2])]
        df_selinf = pd.DataFrame(data=data_selinf.reshape((1, 3)), columns=['coverage', 'risk', 'length'])
        df_selinf['gene_name'] = gene
        df_selinf['num_true_sigs'] = nsignals
        df_selinf['method'] = "Lee_nonrand"
        df_selinf['fdr'] = data[8]
        df_selinf['inferential_power'] = data[7]
        df_selinf['full_risk'] = data[6]

        check_fdr += data[8]
        check_power += data[7]
        check_risk_full += data[6]

        check_power_break[nsignals] += data[7]
        nsig_break[nsignals] += 1.

    df_master = df_master.append(df_selinf, ignore_index=True)

print("count of total files", i)
df_master.to_csv("/Users/snigdhapanigrahi/sim_inference_liver/additional_nonrandomized_inference.csv", index=False)
print("saved to file!")
print("power randomized versus nonrandomized", i, check_risk_full/float(i), check_power/float(i), check_fdr/float(i))
print("power break-up", np.true_divide(check_power_break,nsig_break))