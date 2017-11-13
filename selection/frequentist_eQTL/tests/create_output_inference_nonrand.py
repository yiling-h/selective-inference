import pandas as pd
import numpy as np
import glob, os

# inf_path =r'/Users/snigdhapanigrahi/sim_nonrandomized_inference_liver/nonrand_inference'
#
# allFiles = glob.glob(inf_path + "/*.txt")
# columns = ["coverage", "risk", "length", "gene_name", "num_true_sigs", "method"]
# i = 0
# df_master = pd.DataFrame()
#
# for file_ in allFiles:
#     df = np.loadtxt(file_)
#     gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
#     print("gene", gene)
#
#     i = i + 1
#     data = np.loadtxt(file_)
#     if data.ndim > 1:
#         nactive = data.shape[0]
#     elif data.ndim == 1 and data.shape[0] > 0:
#         nactive = 1.
#     elif data.ndim == 1 and data.shape[0] == 0:
#         nactive = 0.
#
#     if nactive > 1:
#         data_naive = data[:, np.array([1, 5, 3])]
#         nsignals = int(data[0, 6])
#         df_naive = pd.DataFrame(data=data_naive, columns=['coverage', 'risk', 'length'])
#         df_naive = df_naive.assign(gene_name=gene,
#                                    num_true_sigs=nsignals,
#                                    method="naive_nonrand")
#
#         data_selinf = data[:, np.array([0, 4, 2])]
#         df_selinf = pd.DataFrame(data=data_selinf, columns=['coverage', 'risk', 'length'])
#         df_selinf = df_selinf.assign(gene_name=str(gene),
#                                      num_true_sigs=nsignals,
#                                      method="Lee_nonrand")
#
#     elif nactive == 1.:
#         data_naive = data[np.array([1, 5, 3])]
#         nsignals = int(data[6])
#         df_naive = pd.DataFrame(data=data_naive.reshape((1, 3)), columns=['coverage', 'risk', 'length'])
#         df_naive['gene_name'] = gene
#         df_naive['num_true_sigs'] = nsignals
#         df_naive['method'] = "naive_nonrand"
#
#         data_selinf = data[np.array([0, 4, 2])]
#         df_selinf = pd.DataFrame(data=data_selinf.reshape((1, 3)), columns=['coverage', 'risk', 'length'])
#         df_selinf['gene_name'] = gene
#         df_selinf['num_true_sigs'] = nsignals
#         df_selinf['method'] = "Lee_nonrand"
#
#     df_master = df_master.append(df_naive, ignore_index=True)
#     df_master = df_master.append(df_selinf, ignore_index=True)
#
# print("count of total files", i)
# df_master.to_csv("/Users/snigdhapanigrahi/sim_inference_liver/nonrandomized_inference.csv", index=False)
# print("saved to file!")

inf_path =r'/Users/snigdhapanigrahi/inference_liver/Lee_inf/'

df_master = pd.DataFrame()
allFiles = glob.glob(inf_path + "/*.txt")
columns = ["lower_ci", "upper_ci", "point_estimator", "length", "gene_name", "method", "nsignificant", "norm"]
i = 0
check_norm_ad = 0.
check_norm_unad = 0.
check_sig_ad = 0.
check_sig_unad = 0.
check_active = 0.
for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    print("gene", gene)
    if not os.path.exists(os.path.join(inf_path, "Leeoutput_" + str(gene) + ".txt")):
        print("iteration", i)
    else:
        i = i + 1
        data = np.loadtxt(file_)
        if data.ndim > 1:
            nactive = data.shape[0]
        elif data.ndim==1 and data.shape[0]>0:
            nactive = 1.
        elif data.ndim==1 and data.shape[0]==0:
            nactive = 0.

        if nactive > 1:
            data_naive = data[:, np.array([2,3,4])]
            df_naive = pd.DataFrame(data=data_naive, columns=["lower_ci", "upper_ci", "point_estimator"])
            nsig = 0
            for k in range(int(nactive)):
                nsig += 1. - (data[k,2]<0. and data[k,3]>0.)

            nsig = nsig/float(nactive)
            norm = (np.power(data[:, 4], 2).sum())/float(nactive)
            check_norm_unad += norm
            check_sig_unad += nsig
            df_naive = df_naive.assign(length=(data[:, 3] - data[:, 2].sum()) / float(nactive),
                                       gene_name=gene,
                                       method="naive_nonrand",
                                       nsignificant=nsig,
                                       norm=norm)

            data_selinf = data[:, np.array([0, 1, 4])]
            df_selinf = pd.DataFrame(data=data_selinf, columns=["lower_ci", "upper_ci", "point_estimator"])
            nsig_sel = 0
            for k in range(int(nactive)):
                nsig_sel += 1. - (data[k, 0] < 0. and data[k, 1] > 0.)

            nsig_sel = nsig_sel / float(nactive)
            norm_sel = (np.power(data[:, 4], 2).sum()) / float(nactive)
            df_selinf = df_selinf.assign(length = ((data[:,1]- data[:,0]).sum())/float(nactive),
                                         gene_name= str(gene),
                                         method="Lee_nonrand",
                                         nsignificant=nsig_sel,
                                         norm=norm_sel)

            check_norm_ad += norm_sel
            check_sig_ad += nsig_sel
            check_active += nactive

        elif nactive == 1.:
            data_naive = data[np.array([2,3,4])]
            df_naive = pd.DataFrame(data=data_naive.reshape((1,3)), columns=["lower_ci", "upper_ci", "point_estimator"])
            nsig = 1. - (data[2] < 0. and data[3] > 0.)
            norm = np.power(data[4], 2)
            check_norm_unad += norm
            check_sig_unad += nsig
            df_naive['length'] = (data[3] - data[2].sum()) / float(nactive)
            df_naive['gene_name'] = gene
            df_naive['method'] = "naive_nonrand"
            df_naive['nsignificant'] = nsig
            df_naive['norm'] = norm

            data_selinf = data[np.array([0, 1, 4])]
            nsig = 1. - (data[0] < 0. and data[1] > 0.)
            norm = np.power(data[4], 2)
            check_norm_ad += norm
            check_sig_unad += nsig
            df_selinf = pd.DataFrame(data=data_selinf.reshape((1,3)), columns=["lower_ci", "upper_ci", "point_estimator"])
            df_selinf['length'] = data[1] - data[0]
            df_selinf['gene_name'] = gene
            df_selinf['method'] = "Lee_nonrand"
            df_selinf['nsignificant'] = nsig
            df_selinf['norm'] = norm
            check_active += nactive


        df_master = df_master.append(df_naive, ignore_index=True)
        df_master = df_master.append(df_selinf, ignore_index=True)

print("count of total files", i, check_active/float(i))
#df_master.to_csv("/Users/snigdhapanigrahi/inference_liver/real_nonrandomized_inference_0.05.csv", index=False)
print("saved to file!")
