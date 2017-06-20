import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/simes_output_Liver/lasso_results'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):
    snps = np.zeros(len(list_))
    print("number of egenes", len(list_))
    for i in range(len(list_)):

        print("iteration", i)
        snps[i] = list_[i].shape[0]

    return snps
        #print("snps selected", snps)

snps = summary_files(list_)

print(np.mean(snps), np.min(snps), np.max(snps))
print(snps)