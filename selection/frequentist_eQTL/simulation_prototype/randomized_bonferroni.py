from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os
import sys
from scipy.stats import f
from scipy.stats.stats import pearsonr


def bon_selection_egene(X,
                        y,
                        randomizer='gaussian',
                        randomization_scale=1.):
    n, p = X.shape

    T_stats = np.zeros(p)
    for k in range(p):
        T_stats[k] = pearsonr(X[:, k], y)[0]

    if randomizer == 'gaussian':

        perturb = np.random.standard_normal(p)

        randomized_T_stats = np.multiply(np.sign(T_stats), np.sqrt((n - 2) * np.true_divide(T_stats ** 2., 1. - T_stats ** 2.)))\
                             + randomization_scale * perturb

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1. + (randomization_scale ** 2))))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1. + (randomization_scale ** 2))))))

    elif randomizer == 'none':

        randomized_T_stats = (n - 2.) * np.true_divide(T_stats ** 2., 1. - T_stats ** 2.)

        p_val_randomized = np.sort(1. - f.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)), 1, n - 2))

        indices_order = np.argsort(1. - f.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)), 1, n - 2))

    bon_p_randomized = p * p_val_randomized[0]

    t_0 = indices_order[0]

    sigma_hat = np.sqrt((1. - (T_stats[t_0] ** 2)) * np.var(y)) / np.sqrt(n - 2.)

    T_stats_active = T_stats[t_0]

    u = p_val_randomized[1]

    return bon_p_randomized, t_0, indices_order[1], u, np.sign(randomized_T_stats[t_0]), sigma_hat, np.sign(T_stats_active)


if __name__ == "__main__":

    negenes = 500
    path = "/Users/snigdhapanigrahi/selective-inference/selection/frequentist_eQTL/simulation_prototype/data_directory/"
    outdir = "/Users/snigdhapanigrahi/selective-inference/selection/frequentist_eQTL/simulation_prototype/bonferroni_output/"

    iter = int(negenes)

    for j in range(iter):

        output = np.zeros((1, 9))
        X = np.load(os.path.join(path + "X_" + str(j)) + ".npy")
        n, p = X.shape

        y = np.load(os.path.join(path + "y_" + str(j)) + ".npy")
        y = y.reshape((y.shape[0],))

        beta = np.load(os.path.join(path + "b_" + str(j)) + ".npy")
        beta = beta.reshape((beta.shape[0],))

        bon = bon_selection_egene(X, y, randomizer='gaussian', randomization_scale=0.7)
        sys.stderr.write("iteration completed" + str(j) + "\n")
        outfile = os.path.join(outdir, "randomized_bon_" + str(j) + ".txt")

        output[:, 0] = p
        output[:, 1] = beta.sum()
        output[:, 2:] = bon

        np.savetxt(outfile, output)