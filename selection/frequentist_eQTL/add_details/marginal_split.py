from __future__ import print_function
import sys, os
import numpy as np
from selection.algorithms.forward_step import forward_step

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/sim_Egene_data/'

    gene = str("ENSG00000187642.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    print("shape of X", n, p)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X

    sel_risk = 0.
    sel_covered = 0.
    sel_length = 0.

    unad_covered = 0.
    unad_length = 0.
    unad_risk = 0.

    for seed_n in range(100):

        np.random.seed(seed_n)
        y = np.random.standard_normal(n)
        sigma_est = 1.

        subsample_size = int(0.50 * n)

        sel_idx = np.zeros(n, np.bool)
        sel_idx[:subsample_size] = 1
        np.random.shuffle(sel_idx)

        t_test = X[sel_idx,:].T.dot(y[sel_idx])
        index = np.argmax(np.abs(t_test))
        T_sign = np.sign(t_test[index])
        T_observed = (X[~sel_idx,:].T.dot(y[~sel_idx]))[index]
        indicator = np.zeros(p, dtype=bool)
        indicator[index] = 1

        X_inf = X[~sel_idx,:]
        sd = 1./np.sqrt(((X_inf[:,index].T.dot(X_inf[:,index]))))
        unad_intervals = np.array([T_observed - 1.65 * sd, T_observed + 1.65 * sd])
        if (unad_intervals[0] <= 0.) and (0. <= unad_intervals[1]):
            unad_covered += 1

        unad_length += (unad_intervals[1] - unad_intervals[0])
        unad_risk += (T_observed) ** 2.
        print("iteration", seed_n)
    print("metrics", unad_covered /100., unad_length /100., unad_risk/100.)