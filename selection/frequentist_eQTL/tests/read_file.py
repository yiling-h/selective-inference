import glob
import os, numpy as np, pandas, statsmodels.api as sm

#path =r'/Users/snigdhapanigrahi/Results_freq_EQTL/sparsity_5/dim_1/dim_1'
path =r'/Users/snigdhapanigrahi/Results_freq_EQTL/sparsity_0/level_1'

allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)


def evaluation_per_file(list,s,snr =5.):

    if list.ndim == 1:
        list = list.reshape((list.shape[0], 1)).T
    sel_covered = list[:,0]
    sel_length = list[:,1]
    pivots = list[:,2]
    naive_covered = list[:,3]
    naive_pvals = list[:,4]
    naive_length = list[:,5]
    active_set = (list[:,6]).astype(int)
    discoveries = list[:,7]
    ndiscoveries = discoveries.sum()

    nactive = sel_covered.shape[0]
    adjusted_coverage = float(sel_covered.sum() / nactive)
    unadjusted_coverage = float(naive_covered.sum() / nactive)

    adjusted_lengths = float(sel_length.sum() / nactive)
    unadjusted_lengths = float(naive_length.sum() / nactive)

    false_discoveries = 0.
    true_discoveries = 0.
    for i in range(nactive):
        if discoveries[i]>0.:
            if active_set[i]<s:
                true_discoveries += 1.
            else:
                false_discoveries += 1.

    FDR = false_discoveries / max(ndiscoveries, 1.)
    if s>0:
        power = true_discoveries / float(s)
    else:
        power = 0.

    return adjusted_coverage, unadjusted_coverage, adjusted_lengths, unadjusted_lengths, FDR, power

def summary_files(list_):

    coverage_ad = 0.
    coverage_unad = 0.
    length_ad = 0.
    length_unad = 0.
    FDR = 0.
    power = 0.
    length = len(list_)
    print("number of simulations", length)

    for i in range(length):
        print("iteration", i)
        results = evaluation_per_file(list_[i], s=3, snr=5.)
        coverage_ad += results[0]
        coverage_unad += results[1]
        length_ad += results[2]
        length_unad += results[3]
        FDR += results[4]
        power += results[5]

    return coverage_ad / length, coverage_unad / length, length_ad/length, length_unad/length, FDR / length, power / length

print(summary_files(list_))










