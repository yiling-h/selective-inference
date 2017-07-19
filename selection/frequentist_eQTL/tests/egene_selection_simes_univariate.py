from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os
import sys
import regreg.api as rr
import statsmodels.api as sm

def simes_selection_egene(X,
                          y,
                          randomizer= 'gaussian',
                          randomization_scale=1.):

    n, p = X.shape

    sigma_hat = np.zeros(p)

    for k in range(p):

        ols_fit = sm.OLS(y, X[:,k]).fit()
        sigma_hat[k] = np.linalg.norm(ols_fit.resid) / np.sqrt(n - 2.)

    T_stats = np.true_divide(X.T.dot(y),sigma_hat)

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)
        randomized_T_stats = T_stats + randomization_scale * perturb

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

    elif randomizer == 'none':

        randomized_T_stats = T_stats

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)))))

    simes_p_randomized = np.min((p / (np.arange(p) + 1.)) * p_val_randomized)

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

    path = sys.argv[1]
    outdir = sys.argv[2]
    result = sys.argv[3]

    outfile = os.path.join(outdir, "part1_simes_output_sigma_estimated_"+ str(result) + ".txt")

    gene_file = path + "Genes.txt"

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    sys.stderr.write("length" + str(len(content)) + "\n")

    iter = int(len(content))
    output = np.zeros((iter, 7))

    for j in range(iter):

        X = np.load(os.path.join(path + "X_" + str(content[j])) + ".npy")
        n, p = X.shape
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))

        y = np.load(os.path.join(path + "y_" + str(content[j])) + ".npy")
        y = y.reshape((y.shape[0],))

        sys.stderr.write("iteration completed" + str(j) + "\n")
        simes = simes_selection_egene(X, y, randomizer='none')

        output[j, 0] = p
        output[j, 1:] = simes

        #beta = np.load(os.path.join(path + "b_" + str(content[j])) + ".npy")

    np.savetxt(outfile, output)