from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os
import sys

def simes_selection_egene(X,
                          y,
                          randomizer= 'gaussian',
                          noise_level = 1.,
                          randomization_scale=0.31):

    n, p = X.shape
    sigma = noise_level

    T_stats = X.T.dot(y) / sigma

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)
        randomized_T_stats = T_stats + randomization_scale * perturb

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

    elif randomizer == 'none':
        perturb = np.zeros(p)
        randomized_T_stats = T_stats + randomization_scale * perturb

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

    outfile = os.path.join(outdir, "simes_output_rand_0.10_"+ str(result) + ".txt")

    gene_file = path + "Genes.txt"

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    print("length", len(content))

    #output = np.zeros((len(content), 8))
    output = np.zeros((len(content), 7))

    for j in range(len(content)):

        X = np.load(os.path.join(path + "X_" + str(content[j])) + ".npy")
        n, p = X.shape
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))

        y = np.load(os.path.join(path + "y_" + str(content[j])) + ".npy")
        y = y.reshape((y.shape[0],))

        #beta = np.load(os.path.join(path + "b_" + str(content[j])) + ".npy")

        # run Simes
        simes = simes_selection_egene(X, y, randomizer='gaussian')

        output[j, 0] = p
        #output[j, 1] = np.sum(beta > 0.01)
        output[j, 1:] = simes

    np.savetxt(outfile, output)

