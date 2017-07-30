from __future__ import print_function
import sys
import os
from selection.algorithms.lasso import lasso

import numpy as np
import regreg.api as rr
from selection.frequentist_eQTL.estimator import M_estimator_exact
from selection.api import randomization

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_bon_egenes/Egene_data/'

    gene = str("ENSG00000215915.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X


    prototypes = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/Test_bon_egenes/Egene_data/protoclust_" + gene) + ".txt",
                            delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]
    n, p = X.shape

    y = np.load(os.path.join(path + "y_" + gene) + ".npy")
    y = y.reshape((y.shape[0],))
    #sigma_est =  0.3234533
    #sigma_est = 0.5526097
    sigma_est = 0.4303074
    y /= sigma_est

    ####randomized Lasso inference
    np.random.seed(0)
    lam = np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    epsilon = 1. / np.sqrt(n)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=1.)
    loss = rr.glm.gaussian(X, y)

    M_est = M_estimator_exact(loss, epsilon, penalty, randomization)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed) + "\n")

    np.random.seed(0)
    lam_frac = .8
    lam = lam_frac* np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()

    active_lasso = soln != 0
    nactive_lasso = active_lasso.sum()
    print("nactive", nactive_lasso)

    active_set_lasso = np.nonzero(active_lasso)[0]
    print("active set", active_set_lasso)

    print("intersected set", np.intersect1d(active_set_lasso, active_set))
