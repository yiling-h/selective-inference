import numpy as np, os, itertools, sys
import pandas as pd
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import lasso_full
from scipy.stats import norm as ndist

from selection.adjusted_MLE.cv_MLE import (sim_xy,
                                           selInf_R,
                                           glmnet_lasso,
                                           BHfilter,
                                           coverage)

def pivot(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.15, target= "selected",
          randomizer_scale=np.sqrt(0.50), full_dispersion=True):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    _, _, _, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory / float(n))

    randomized_lasso = lasso.gaussian(X,
                                      y,
                                      feature_weights=lam_theory * np.ones(p),
                                      randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())],
                                  np.bool)


pivot()