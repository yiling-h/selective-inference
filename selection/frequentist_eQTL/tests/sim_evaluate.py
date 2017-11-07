import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/sim_inference_liver/inference0'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)

def summary_files(list_):

    coverage_ad = 0.
    coverage_unad = 0.

    risk_ad = 0.
    risk_unad = 0.

    length_ad = 0.
    length_unad = 0.

    power = 0.
    fdr = 0.

    retry = 0.

    length = len(list_)
    print("number of simulations", length)

    for i in range(length):
        print("iteration", i)

        results = list_[i]

        if results.ndim > 1:
            nactive = results.shape[0]
        elif results.ndim==1 and results.shape[0]>0:
            nactive = 1.
        elif results.ndim==1 and results.shape[0]==0:
            nactive = 0.

       # print("results", results, results.shape)
        if nactive > 1 and results[:, 8].sum() / float(nactive) < 20.:
            coverage_ad += results[:, 6].sum() / float(nactive)
            coverage_unad += results[:, 7].sum() / float(nactive)

            risk_ad += results[:, 8].sum() / float(nactive)
            risk_unad += results[:, 9].sum() / float(nactive)

            length_ad += results[:, 10].sum() / float(nactive)
            length_unad += results[:, 11].sum() / float(nactive)

            power += (results[:, 13])[0]
            fdr += (results[:, 14])[0]

            if (results[:, 15][0] == 0):
                retry += 1

            print("power, fdr", (results[:, 13])[0], (results[:, 14])[0])
        elif nactive == 1. and results[8].sum() / float(nactive) < 20.:
            coverage_ad += results[6]
            coverage_unad += results[7]

            risk_ad += results[8]
            risk_unad += results[9]

            length_ad += results[10]
            length_unad += results[11]

            power += results[13]
            fdr += results[14]
            print("power, fdr", results[13], results[14])
            if (results[15] == 0):
                retry += 1
        else:
            length = length - 1.

    print("length", length)
    return coverage_ad / length, coverage_unad / length, risk_ad /length, risk_unad /length, \
           length_ad/length, length_unad/length, fdr/length, power/length, retry

print("results", summary_files(list_))