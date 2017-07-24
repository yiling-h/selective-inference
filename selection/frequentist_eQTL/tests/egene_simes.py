from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os
import sys
import regreg.api as rr
import statsmodels.api as sm
from scipy.stats import f
from scipy.stats.stats import pearsonr

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def simes_selection_egene(X,
                          y,
                          randomizer= 'gaussian',
                          randomization_scale=1.):

    n, p = X.shape

    # sigma_hat = np.zeros(p)
    #
    # for k in range(p):
    #
    #     ols_fit = sm.OLS(y, X[:,k]).fit()
    #     sigma_hat[k] = np.linalg.norm(ols_fit.resid) / np.sqrt(n - 2.)
    #
    # T_stats = np.true_divide(X.T.dot(y),sigma_hat)
    T_stats = np.zeros(p)
    for k in range(p):
        T_stats[k] = pearsonr(X[:,k], y)[0]

    #print("corr", T_stats[5205])

    if randomizer == 'gaussian':

        perturb = np.random.standard_normal(p)
        randomized_T_stats = T_stats + randomization_scale * perturb

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

    elif randomizer == 'none':

        randomized_T_stats = (n-2.)* np.true_divide(T_stats**2., np.ones(p)-T_stats**2.)

        p_val_randomized = np.sort(1. - f.cdf(randomized_T_stats,1, n-2))

        indices_order = np.argsort(1. - f.cdf(randomized_T_stats,1, n-2))

        #randomized_T_stats = T_stats

        #p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)))))

        #indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)))))
    print("p randomized", p_val_randomized[:20])
    print("indices", indices_order[:20])

    simes_p_randomized = np.min((p / (np.arange(p) + 1.)) * p_val_randomized)

    print("simes p randomized", ((p / (np.arange(p) + 1.)) * p_val_randomized)[:20])

    i_0 = np.argmin((p / (np.arange(p) + 1.)) * p_val_randomized)

    t_0 = indices_order[i_0]

    T_stats_active = T_stats[i_0]

    u_1 = ((i_0 + 1.) / p) * np.min(
        np.delete((p / (np.arange(p) + 1.)) * p_val_randomized, i_0))

    if i_0 > p - 2:
        u_2 = -1
    else:
        u_2 = p_val_randomized[i_0 + 1]

    return simes_p_randomized, i_0, t_0, np.sign(T_stats_active), u_1, u_2

if __name__ == "__main__":

    #np.random.seed(2)
    path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

    X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
    #X = np.load(os.path.join(path + "X_" + "ENSG00000218510.3") + ".npy")

    #X_transposed = unique_rows(X.T)
    #X = X_transposed.T
    n, p = X.shape
    print("dims", n,p)

    y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
    #y = np.load(os.path.join(path + "y_" + "ENSG00000218510.3") + ".npy")
    y = y.reshape((y.shape[0],))

    simes = simes_selection_egene(X, y, randomizer='none')

    print("simes output", simes)
