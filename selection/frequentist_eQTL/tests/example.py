import pandas as pd
import numpy as np

method = "sel_inf"
allFiles = ["/Users/snigdhapanigrahi/sim_inference_liver/inference0/inference_ENSG00000001629.5.txt",
            "/Users/snigdhapanigrahi/sim_inference_liver/inference0/inference_ENSG00000000938.8.txt"]

columns = ["gene_name", "num_true_sigs", "risk", "coverage", "length", "method"]

df_master = pd.DataFrame()

for file_ in allFiles:
    gene = file_.strip().split('/')[-1].split(".txt")[0].split("_")[1]
    n_true_sigs = 0  # TODO: figure this out

    print(file_)
    data = np.loadtxt(file_)

    df_naive = pd.DataFrame(data=data[:,:3], columns=['coverage', 'risk', 'length'])  # TODO
    print(df_naive)
    df_naive = df_naive.assign(gene_name=gene,
                               n_true_sigs=n_true_sigs,
                               method="naive")
    # make sure the methods are the right name
    print(df_naive)
    # df_selinf
    df_selinf = pd.DataFrame(data=data[:,:4])  # TODO
    df_selinf = df_selinf.assign(gene_name=gene,
                                 n_true_sigs=n_true_sigs,
                                 method="sel_inf")
    print(df_selinf)

    df_master = df_master.append(df_naive, ignore_index=True)
    df_master = df_master.append(df_selinf, ignore_index=True)

print(df_master)

df_master.to_csv("results.csv", index=False)
print("saved to file!")

# to load data
df_loaded = pd.read_csv("results.csv")

print(df_loaded)