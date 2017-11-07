import pandas as pd
import numpy as np
import glob, os
import pandas as pd

fwdbwd_path = r'/Users/snigdhapanigrahi/inference_liver/sel_SNPs_with_MAFs_fwdbwd/'
fwd_path = r'/Users/snigdhapanigrahi/fwd_bwd_inference/inference0'
adjusted_path = r'/Users/snigdhapanigrahi/inference_liver/sel_SNPs_with_MAFs/'
inf_path = r'/Users/snigdhapanigrahi/inference_liver/inference0'

df_master = pd.DataFrame()
allFiles_adjusted = glob.glob(inf_path + "/*.txt")
columns = ["SNP_index", "eVariant", "gene_name", "effect_size","method"]

for file_ in allFiles_adjusted:
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]

    if not os.path.exists(os.path.join(adjusted_path, "sel_SNPs_MAFs_" + str(gene) + ".txt")) or \
            not os.path.exists(os.path.join(inf_path, "inference_" + str(gene) + ".txt")):
        print("gene", gene)
    else:
        sel_MAF = pd.read_csv(os.path.join(adjusted_path, "sel_SNPs_MAFs_" + str(gene) + ".txt"), sep="\t", header=None)
        sel_MAF.columns = ["SNP_index", "eVariant", "MAF"]
        # print("sel MAF", sel_MAF)
        data = np.loadtxt(file_)

        if data.ndim > 1:
            nactive = data.shape[0]
            find = np.zeros(int(nactive), np.bool)
            #print("true false", data[:,1]<0., data[:,0]>0.)
            find[(data[:,1]<0.)] = 1
            find[(data[:,0]>0.)] = 1
            find_sel = pd.Series(find, name='bools')
        elif data.ndim == 1 and data.shape[0] > 0:
            nactive = 1.
            find = np.zeros(1, np.bool)
            find[0] = (data[1]<0. or data[0] > 0.)
        elif data.ndim == 1 and data.shape[0] == 0:
            nactive = 0.

        if nactive > 1:
            df_naive = sel_MAF[find_sel.values]
            df_naive = df_naive.assign(effect_size=data[:, 4][find],
                                       gene_name=gene,
                                       method="sel_inf")

            # print("df naive", df_naive)

        elif nactive == 1. and find[0]== True:
            df_naive = sel_MAF
            df_naive = df_naive.assign(effect_size=data[4],
                                       gene_name=gene,
                                       method="sel_inf")

            # print("df naive", df_naive)

        df_master = df_master.append(df_naive, ignore_index=True)

df_master.to_csv("/Users/snigdhapanigrahi/inference_liver/effect_sizes_adjusted.csv", index=False)

# df_master = pd.DataFrame()
# allFiles_adjusted = glob.glob(fwd_path + "/*.txt")
# columns = ["SNP_index", "eVariant", "gene_name", "effect_size","method"]
#
# for file_ in allFiles_adjusted:
#     gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[3]
#
#     if not os.path.exists(os.path.join(fwdbwd_path, "sel_SNPs_MAFs_" + str(gene) + ".txt")) or \
#             not os.path.exists(os.path.join(fwd_path, "fwd_bwd_output_" + str(gene) + ".txt")):
#         print("gene", gene)
#     else:
#         sel_MAF = pd.read_csv(os.path.join(fwdbwd_path, "sel_SNPs_MAFs_" + str(gene) + ".txt"), sep="\t", header=None)
#         sel_MAF.columns = ["SNP_index", "eVariant", "MAF"]
#         # print("sel MAF", sel_MAF)
#         data = np.loadtxt(file_)
#
#         if data.ndim > 1:
#             nactive = data.shape[0]
#         elif data.ndim == 1 and data.shape[0] > 0:
#             nactive = 1.
#         elif data.ndim == 1 and data.shape[0] == 0:
#             nactive = 0.
#
#         if nactive > 1:
#             df_naive = sel_MAF
#             df_naive = df_naive.assign(effect_size=data[:, 4],
#                                        gene_name=gene,
#                                        method="fwd_bwd")
#             # print("df naive", df_naive)
#
#         elif nactive == 1.:
#             df_naive = sel_MAF
#             df_naive = df_naive.assign(effect_size=data[4],
#                                        gene_name=gene,
#                                        method="fwd_bwd")
#             # print("df naive", df_naive)
#
#         df_master = df_master.append(df_naive, ignore_index=True)
#
# df_master.to_csv("/Users/snigdhapanigrahi/inference_liver/effect_sizes_fwd_bwd.csv", index=False)


