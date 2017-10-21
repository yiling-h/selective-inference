from __future__ import print_function
import sys
import os

from selection.algorithms.lasso import lasso
import numpy as np
from scipy.stats.stats import pearsonr


def lasso_Gaussian(X, y, lam, true_mean, signal_indices):
    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()
    active = soln != 0
    print("Lasso estimator", soln[active])
    nactive = active.sum()
    print("nactive", nactive)

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
    true_val = projection_active.T.dot(true_mean)

    active_set = np.nonzero(active)[0]
    print("active set", active_set)

    active_signs = np.sign(soln[active])
    C = L.constraints
    coverage_unad = np.zeros(nactive)

    if C is not None:
        one_step = L.onestep_estimator
        print("one step", one_step)
        point_est = projection_active.T.dot(y)
        sd = np.sqrt(np.linalg.inv(X[:, active].T.dot(X[:, active])).diagonal())
        unad_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T
        unad_length = (unad_intervals[:, 1] - unad_intervals[:, 0]).sum() / nactive
        unad_risk = np.power(point_est - true_val, 2.).sum() / nactive

        true_indices = (signal_indices.nonzero())[0]

        for k in range(one_step.shape[0]):
            if (unad_intervals[k, 0] <= true_val[k]) and (true_val[k] <= unad_intervals[k, 1]):
                coverage_unad[k] += 1
        coverage_unad = coverage_unad.sum() / nactive

        corr_nonrand = np.zeros(true_indices.shape[0])

        true_sel = []

        if nactive > 1:
            for j in range(nactive):
                if true_indices.shape[0] >= 1:
                    for l in range(true_indices.shape[0]):
                        corr_nonrand[l] = pearsonr(X[:, active_set[j]], X_unpruned[:, true_indices[l]])[0]
                    if np.any(corr_nonrand >= 0.49):
                        true_sel.append(active_set[j])
        elif nactive == 1:
            if true_indices.shape[0] >= 1:
                for l in range(true_indices.shape[0]):
                    corr_nonrand[l] = pearsonr(X[:, active_set[0]], X_unpruned[:, true_indices[l]])[0]
                if np.any(corr_nonrand >= 0.49):
                    true_sel.append(active_set[0])

        power_nonrand = true_sel.shape[0] / max(1., float(true_indices.shape[0]))
        return np.vstack((true_indices.shape[0], power_nonrand, coverage_unad, unad_length, unad_risk))

    else:

        return np.vstack((0., 0., 0., 0., 0.))


if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    for egene in range(len(content)):
        gene = str(content[egene])
        try:
            X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
            n, p = X.shape
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))
            X_unpruned = X

            beta = np.load(os.path.join(inpath + "b_" + gene) + ".npy")
            beta = beta.reshape((beta.shape[0],))
            beta = np.sqrt(n) * beta
            true_mean = X_unpruned.dot(beta)
            signal_indices = np.abs(beta) > 0.005

            prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
            prototypes = np.unique(prototypes).astype(int)
            print("prototypes", prototypes.shape[0])
            X = X[:, prototypes]

            y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
            y = y.reshape((y.shape[0],))

            sigma_est = 1.
            y /= sigma_est

            lam_frac = 1.
            lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

            lasso_results = lasso_Gaussian(X,
                                           y,
                                           lam,
                                           true_mean,
                                           signal_indices)

            outfile = os.path.join(outdir + "Leeoutput_" + gene + ".txt")
            np.savetxt(outfile, lasso_results)
            sys.stderr.write("Iteration completed" + str(egene) + "\n")

        except:
            sys.stderr.write("Error" + str(egene) + "\n")

