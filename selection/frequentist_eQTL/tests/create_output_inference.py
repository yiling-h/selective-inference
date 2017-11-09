import pandas as pd
import numpy as np
import glob, os

# inf_path =r'/Users/snigdhapanigrahi/sim_inference_liver/inference0'
# sel_path =r'/Users/snigdhapanigrahi/sim_inference_liver/nonrand_sel'
# df_master = pd.DataFrame()
#
# allFiles = glob.glob(inf_path + "/*.txt")
# columns = ["coverage", "risk", "length", "gene_name", "num_true_sigs", "method", "power_rand", "power_nonrand"]
# i = 0
# check_prand = 0.
# check_pnonrand = 0.
#
# for file_ in allFiles:
#     df = np.loadtxt(file_)
#     gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
#     print("gene", gene)
#     if not os.path.exists(os.path.join(inf_path, "inference_" + str(gene) + ".txt")):
#         print("iteration", i)
#     else:
#         i = i + 1
#         data = np.loadtxt(file_)
#         if data.ndim > 1:
#             nactive = data.shape[0]
#         elif data.ndim==1 and data.shape[0]>0:
#             nactive = 1.
#         elif data.ndim==1 and data.shape[0]==0:
#             nactive = 0.
#
#         if nactive > 1:
#             data_naive = data[:, np.array([7, 9, 11])]
#             sel = np.loadtxt(os.path.join(sel_path, "nonrand_sel_" + str(gene) + ".txt"))
#             nsignals = int(sel[6])
#             power_nonrand = sel[10]
#             power_rand = sel[9]
#             check_prand += power_rand
#             check_pnonrand += power_nonrand
#
#             df_naive = pd.DataFrame(data=data_naive, columns=['coverage', 'risk', 'length'])
#             #print(df_naive)
#             df_naive = df_naive.assign(gene_name = gene,
#                                        num_true_sigs=nsignals,
#                                        method="naive",
#                                        power_nonrand = power_nonrand,
#                                        power_rand = power_rand)
#
#             data_selinf = data[:, np.array([6, 8, 10])]
#             df_selinf = pd.DataFrame(data=data_selinf, columns=['coverage', 'risk', 'length'])
#             df_selinf = df_selinf.assign(gene_name= str(gene),
#                                          num_true_sigs=nsignals,
#                                          method="sel_inf",
#                                          power_nonrand=power_nonrand,
#                                          power_rand=power_rand)
#
#         elif nactive == 1.:
#             data_naive = data[np.array([7, 9, 11])]
#             #print("shape", data.shape, data_naive.shape)
#             sel = np.loadtxt(os.path.join(sel_path, "nonrand_sel_" + str(gene) + ".txt"))
#             nsignals = int(sel[6])
#             power_nonrand = sel[10]
#             power_rand = sel[9]
#
#             check_prand += power_rand
#             check_pnonrand += power_nonrand
#
#             #print(data_naive)
#             df_naive = pd.DataFrame(data=data_naive.reshape((1,3)), columns=['coverage', 'risk', 'length'])
#             #print(df_naive)
#             df_naive['gene_name'] = gene
#             df_naive['num_true_sigs'] = nsignals
#             df_naive['method'] = "naive"
#             df_naive['power_nonrand'] = power_nonrand
#             df_naive['power_rand'] = power_rand
#             #df_naive = df_naive.assign(gene_name=gene,
#             #                           num_true_sigs=nsignals,
#             #                           method="naive")
#             print(df_naive)
#
#             data_selinf = data[np.array([6, 8, 10])]
#             print("shape", data_selinf.shape)
#             df_selinf = pd.DataFrame(data=data_selinf.reshape((1,3)), columns=['coverage', 'risk', 'length'])
#             df_selinf['gene_name'] = gene
#             df_selinf['num_true_sigs'] = nsignals
#             df_selinf['method'] = "sel_inf"
#             df_selinf['power_nonrand'] = power_nonrand
#             df_selinf['power_rand'] = power_rand
#             #df_selinf = df_selinf.assign(gene_name=gene,
#             #                             num_true_sigs=nsignals,
#             #                             method="sel_inf")
#
#         df_master = df_master.append(df_naive, ignore_index=True)
#         df_master = df_master.append(df_selinf, ignore_index=True)

#print("count of total files", i)
#df_master.to_csv("/Users/snigdhapanigrahi/sim_inference_liver/adjusted_unadjusted_inference.csv", index=False)
#print("saved to file!")
#print("power randomized versus nonrandomized", check_prand/1764., check_pnonrand/1764.)

inf_path =r'/Users/snigdhapanigrahi/inference_liver/inference_05'
df_master = pd.DataFrame()
allFiles = glob.glob(inf_path + "/*.txt")
columns = ["lower_ci", "upper_ci", "point_estimator", "length", "gene_name", "method", "nsignificant", "norm"]
i = 0
check_norm_ad = 0.
check_norm_unad = 0.
check_sig_ad = 0.
check_sig_unad = 0.
check_length = 0.
for file_ in allFiles:
    df = np.loadtxt(file_)
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    print("gene", gene)
    if not os.path.exists(os.path.join(inf_path, "inference_" + str(gene) + ".txt")):
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
            data_naive = data[:, np.array([2,3,5])]
            df_naive = pd.DataFrame(data=data_naive, columns=["lower_ci", "upper_ci", "point_estimator"])
            nsig = 0
            for k in range(int(nactive)):
                nsig += 1. - (data[k,2]<0. and data[k,3]>0.)

            nsig = nsig/float(nactive)
            norm = (np.power(data[:, 5], 2).sum())/float(nactive)
            check_norm_unad += norm
            check_sig_unad += nsig
            df_naive = df_naive.assign(length=(data[:, 3] - data[:, 2].sum()) / float(nactive),
                                       gene_name=gene,
                                       method="naive",
                                       nsignificant=nsig,
                                       norm=norm)

            data_selinf = data[:, np.array([0, 1, 4])]
            df_selinf = pd.DataFrame(data=data_selinf, columns=["lower_ci", "upper_ci", "point_estimator"])
            nsig_sel = 0
            for k in range(int(nactive)):
                nsig_sel += 1. - (data[k, 0] < 0. and data[k, 1] > 0.)

            nsig_sel = nsig_sel / float(nactive)
            norm_sel = (np.power(data[:, 4], 2).sum()) / float(nactive)
            check_length += ((data[:,1]- data[:,0]).sum())/float(nactive)
            df_selinf = df_selinf.assign(length = ((data[:,1]- data[:,0]).sum())/float(nactive),
                                         gene_name= str(gene),
                                         method="sel_inf",
                                         nsignificant=nsig_sel,
                                         norm=norm_sel)

            check_norm_ad += norm_sel
            check_sig_ad += nsig_sel

        elif nactive == 1.:
            data_naive = data[np.array([2,3,5])]
            df_naive = pd.DataFrame(data=data_naive.reshape((1,3)), columns=["lower_ci", "upper_ci", "point_estimator"])
            nsig = 1. - (data[2] < 0. and data[3] > 0.)
            norm = np.power(data[5], 2)
            check_norm_unad += norm
            check_sig_unad += nsig
            df_naive['length'] = (data[3] - data[2].sum()) / float(nactive)
            df_naive['gene_name'] = gene
            df_naive['method'] = "naive"
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
            df_selinf['method'] = "sel_inf"
            df_selinf['nsignificant'] = nsig
            df_selinf['norm'] = norm

            check_length += (data[1] - data[0])


        df_master = df_master.append(df_naive, ignore_index=True)
        df_master = df_master.append(df_selinf, ignore_index=True)

print("count of total files", i)
print("norms", check_norm_unad/1761., check_norm_ad/1761., check_sig_unad/1761., check_sig_ad/1761., check_length/1761.)
df_master.to_csv("/Users/snigdhapanigrahi/inference_liver/real_adjusted_unadjusted_inference_0.05.csv", index=False)
print("saved to file!")




