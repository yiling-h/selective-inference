import pandas as pd
import numpy as np
import glob, os

inf_path =r'/Users/snigdhapanigrahi/sim_inference_liver/additional_info_nonrand/additional_info'
fdr_path =r'/Users/snigdhapanigrahi/sim_inference_liver/lasso_fdr_nonrand'

allFiles = glob.glob(inf_path + "/*.txt")
columns = ["coverage", "risk", "length", "gene_name", "num_true_sigs", "method", "fdr", "power", "full_risk"]
i = 0
df_master = pd.DataFrame()

check_fdr = 0.
check_power = 0.
check_risk_full = 0.

check_power_break = np.zeros(10)
check_sel_power_break = np.zeros(10)
check_fdr_break = np.zeros(10)
nsig_break = np.zeros(10)
sel_report = 0.

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
        fdr_file = np.loadtxt(os.path.join(fdr_path, "fdr_" + str(gene) + ".txt"))
        if fdr_file.ndim == 2:
            fdr_screening = fdr_file[0,1]
        else:
            fdr_screening = fdr_file[1]
        print("fdr", fdr_screening)
        nsignals = int(data[0, 9])
        data_selinf = data[:, np.array([0, 4, 2])]

        df_selinf = pd.DataFrame(data=data_selinf, columns=['coverage', 'risk', 'length'])

        if (np.multiply(nactive, 1. - fdr_screening)) > 0. and (nsignals / (nactive * (1. - fdr_screening))) > 1.:
            sel_power = (data[0, 7] * nsignals) / (nactive * (1. - fdr_screening))
        else:
            sel_power = (data[0, 7])
        df_selinf = df_selinf.assign(gene_name=str(gene),
                                     num_true_sigs=nsignals,
                                     method="Lee_nonrand",
                                     fdr=data[0,8],
                                     inferential_power=data[0,7],
                                     full_risk=data[0, 6],
                                     sel_power = sel_power)
        check_fdr += data[0,8]
        check_power += data[0,7]
        check_risk_full += data[0,6]

        check_power_break[nsignals] += data[0,7]

        if (np.multiply(nactive, 1. - fdr_screening)) > 0. and (nsignals / (nactive * (1. - fdr_screening))) > 1.:
            check_sel_power_break[nsignals] += (data[0, 7] * nsignals) / (nactive * (1. - fdr_screening))
        else:
            check_sel_power_break[nsignals] += (data[0, 7])

        nsig_break[nsignals] += 1.
        sel_report += data[0,9]

    elif nactive == 1.:
        fdr_file = np.loadtxt(os.path.join(fdr_path, "fdr_" + str(gene) + ".txt"))
        print("fdr_file", fdr_file, )
        if fdr_file.ndim == 2:
            fdr_screening = fdr_file[0,1]
        else:
            fdr_screening = fdr_file[1]

        print("fdr", fdr_screening)
        nsignals = int(data[9])
        data_selinf = data[np.array([0, 4, 2])]
        df_selinf = pd.DataFrame(data=data_selinf.reshape((1, 3)), columns=['coverage', 'risk', 'length'])

        if (1. - fdr_screening) > 0.:
            sel_power = ((data[7] * nsignals) / (1. - fdr_screening))
        else:
            sel_power = data[7]
        df_selinf['gene_name'] = gene
        df_selinf['num_true_sigs'] = nsignals
        df_selinf['method'] = "Lee_nonrand"
        df_selinf['fdr'] = data[8]
        df_selinf['inferential_power'] = data[7]
        df_selinf['full_risk'] = data[6]
        df_selinf['sel_power'] = sel_power
        #df_selinf['fdr_screening'] = fdr_screening

        check_fdr += data[8]
        check_power += data[7]
        check_risk_full += data[6]

        check_power_break[nsignals] += data[7]
        #check_fdr_break[nsignals] += fdr_screening
        nsig_break[nsignals] += 1.

        sel_report += data[9]

        if (1. - fdr_screening) > 0.:
            check_sel_power_break[nsignals] += ((data[7] * nsignals) / (1. - fdr_screening))
        else:
            check_sel_power_break[nsignals] += data[7]

    df_master = df_master.append(df_selinf, ignore_index=True)

print("count of total files", i)
print("power break-up", np.true_divide(check_power_break,nsig_break),
      np.true_divide(check_sel_power_break,nsig_break))