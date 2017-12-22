import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/sim_inference_liver/additional_info'
inf0_path =r'/Users/snigdhapanigrahi/sim_inference_liver/inference0'
sel_path =r'/Users/snigdhapanigrahi/sim_inference_liver/nonrand_sel'
df_master = pd.DataFrame()

allFiles = glob.glob(inf_path + "/*.txt")
columns = ["coverage", "risk", "full_risk", "full_naive_risk",
           "length", "gene_name", "num_true_sigs", "method", "fdr",
           "inferential_power"]
i = 0
check_risk_sel = 0.
check_risk_naive = 0.
check_power = 0.
check_fdr = 0.

check_power_break = np.zeros(10)
nsig_break = np.zeros(10)
nextreme = 0.

for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    print("gene", gene)
    if os.path.exists(os.path.join(inf0_path, "inference_" + str(gene) + ".txt")):
        i = i + 1
        data = np.loadtxt(file_)
        if data.ndim > 1:
            nactive = data.shape[0]
        elif data.ndim == 1 and data.shape[0] > 0:
            nactive = 1.
        elif data.ndim == 1 and data.shape[0] == 0:
            nactive = 0.

        if nactive > 1:
            sel = np.loadtxt(os.path.join(sel_path, "nonrand_sel_" + str(gene) + ".txt"))
            nsignals = int(sel[6])
            data_selinf = data[:, np.array([6, 8, 10])]
            df_selinf = pd.DataFrame(data=data_selinf, columns=['coverage', 'risk', 'length'])
            df_selinf = df_selinf.assign(gene_name=str(gene),
                                         num_true_sigs=nsignals,
                                         method="sel_inf",
                                         fdr=data[0, 14],
                                         inferential_power=data[0, 13],
                                         full_risk=data[0, 15],
                                         full_naive_risk=data[0, 16])

            check_risk_sel += data[0, 15]
            check_risk_naive += data[0, 16]
            check_power += data[0, 13]
            check_fdr += data[0, 14]

            check_power_break[nsignals] += data[0, 13]
            nsig_break[nsignals] += 1.

        elif nactive == 1.:
            sel = np.loadtxt(os.path.join(sel_path, "nonrand_sel_" + str(gene) + ".txt"))
            nsignals = int(sel[6])
            data_selinf = data[np.array([6, 8, 10])]
            print("shape", data_selinf.shape)
            df_selinf = pd.DataFrame(data=data_selinf.reshape((1, 3)), columns=['coverage', 'risk', 'length'])
            df_selinf['gene_name'] = gene
            df_selinf['num_true_sigs'] = nsignals
            df_selinf['method'] = "sel_inf"
            df_selinf['fdr'] = data[14]
            df_selinf['inferential_power'] = data[13]
            df_selinf['full_risk'] = data[15]
            df_selinf['full_naive_risk'] = data[16]

            check_risk_sel += data[15]
            check_risk_naive += data[16]
            check_power += data[13]
            check_fdr += data[14]

            check_power_break[nsignals] += data[13]
            nsig_break[nsignals] += 1.

        df_master = df_master.append(df_selinf, ignore_index=True)

    else:
        print("iteration", i)


print("count of total files", i)
df_master.to_csv("/Users/snigdhapanigrahi/sim_inference_liver/additional_adjusted_inference.csv", index=False)
print("saved to file!")
print("power randomized versus nonrandomized", i, check_risk_sel/float(i), check_risk_naive/float(i),
      check_power/float(i), check_fdr/float(i), nextreme)
print("power break-up", np.true_divide(check_power_break,nsig_break))