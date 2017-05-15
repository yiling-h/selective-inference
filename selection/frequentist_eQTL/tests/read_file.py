import glob
import os, numpy as np, pandas, statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import probplot, uniform


#path =r'/Users/snigdhapanigrahi/Results_freq_EQTL/sparsity_5/dim_1/dim_1'
path =r'/Users/snigdhapanigrahi/Results_freq_EQTL/high_dim_test'

allFiles = glob.glob(path + "/*.txt")

list_ = []
for file_ in allFiles:
    df = np.loadtxt(file_)
    list_.append(df)


def evaluation_per_file(list,s):

    if list.ndim == 1:
        list = list.reshape((list.shape[0], 1)).T
    sel_covered = list[:,0]
    sel_length = list[:,1]
    pivots = list[:,2]
    naive_covered = list[:,3]
    naive_pvals = list[:,5]
    naive_length = list[:,6]
    active_set = (list[:,7]).astype(int)
    discoveries = list[:,8]
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

    return adjusted_coverage, unadjusted_coverage, adjusted_lengths, unadjusted_lengths, FDR, power, pivots, naive_pvals

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
        results = evaluation_per_file(list_[i], s=5)
        coverage_ad += results[0]
        coverage_unad += results[1]
        length_ad += results[2]
        length_unad += results[3]
        FDR += results[4]
        power += results[5]


    return coverage_ad / length, coverage_unad / length, length_ad/length, length_unad/length, FDR / length, power / length

print(summary_files(list_))

def plot_p_values():

    length = len(list_)
    print("number of simulations", length)

    selective_pivots = []
    naive_pivots = []

    for i in range(length):
        print("iteration", i)
        results = evaluation_per_file(list_[i], s=5)

        #print(results[6], results[7])
        selective_pivots = np.concatenate((selective_pivots, results[6]), axis=0)
        naive_pivots = np.concatenate((naive_pivots, results[7]), axis=0)

    coverage = True
    color = 'b'
    label = None

    fig = plt.figure()

    ax = fig.gca()

    fig.suptitle('Selective and naive pivots')

    #print("sel pivots", selective_pivots)

    ecdf = sm.distributions.ECDF(selective_pivots)

    G = np.linspace(0, 1)
    F_pivot = ecdf(G)

    ax.plot(G, F_pivot, '-o', c=color, lw=2, label="Selective pivots")
    ax.plot([0, 1], [0, 1], 'k-', lw=2)

    ecdf_naive = sm.distributions.ECDF(naive_pivots)

    F_naive = ecdf_naive(G)
    ax.plot(G, F_naive, '-o', c='r', lw=2, label="Naive pivots")
    ax.plot([0, 1], [0, 1], 'k-', lw=2)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='lower right')

    plt.savefig('/Users/snigdhapanigrahi/Documents/Research/Python_plots/p_val.pdf', bbox_inches='tight')


#plot_p_values()






