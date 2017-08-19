import glob
import os, numpy as np, pandas, statsmodels.api as sm

path =r'/Users/snigdhapanigrahi/sim_hs_inference/inference'

allFiles = glob.glob(path + "/*.txt")
list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)
    #if 'inference_lrs_' in file_:


def summary_files(list_):

    coverage_ad = 0.
    coverage_unad = 0.

    risk_ad = 0.
    risk_unad = 0.

    length_ad = 0.
    length_unad = 0.

    power = 0.
    fdr = 0.

    length = len(list_)
    print("number of simulations", length)

    for i in range(length):
        print("iteration", i)
        results = list_[i]
        nactive = results.shape[0]

        coverage_ad += results[:,6].sum()/float(nactive)
        coverage_unad += results[:,7].sum()/float(nactive)

        risk_ad += results[:,8].sum()/float(nactive)
        risk_unad += results[:,9].sum()/float(nactive)

        length_ad += results[:,10].sum()/float(nactive)
        length_unad += results[:,11].sum()/float(nactive)

        power += (results[:,13])[0]
        fdr += (results[:,13])[1]

    return coverage_ad / length, coverage_unad / length, risk_ad /length, risk_unad /length, \
           length_ad/length, length_unad/length, power/length, fdr/length

print("results", summary_files(list_))