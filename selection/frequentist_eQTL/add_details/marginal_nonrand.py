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
    count = 0.

    for seed_n in range(100):

        np.random.seed(seed_n)
        y = np.random.standard_normal(n)
        sigma_est = 1.

        t_test = X.T.dot(y)
        index = np.argmax(np.abs(t_test))
        T_sign = np.sign(t_test[index])
        T_observed = (X.T.dot(y))[index]
        indicator = np.zeros(p, dtype=bool)
        indicator[index] = 1

        sd = 1.
        unad_intervals = np.array([T_observed - 1.65 * sd, T_observed + 1.65 * sd])
        if (unad_intervals[0] <= 0.) and (0. <= unad_intervals[1]):
            unad_covered += 1

        unad_length += (unad_intervals[1] - unad_intervals[0])

        try:
            FS = forward_step(X, y, covariance=np.identity(n))
            FS.step()
            intervals = FS.model_pivots(1,
                                        alternative='twosided',
                                        saturated=False,
                                        ndraw=5000,
                                        burnin=2000,
                                        which_var=[],
                                        compute_intervals=True,
                                        nominal=False,
                                        coverage=0.90)

            sel_risk += t_test[index]** 2.
            sel_int = ((intervals[1])[0])[1]
            if sel_int[0] <0. and sel_int[1]>0.:
                sel_covered +=1
            sel_length += sel_int[1]-sel_int[0]
        except ValueError:
            count += 1
        print("iteration", seed_n)
    print("count", count, sel_risk/(100. -count), sel_covered/(100.-count), sel_length/(100.-count),
          unad_covered/(100.-count), unad_length/(100.-count))
