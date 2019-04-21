import numpy as np
import nose.tools as nt

from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.tests.instance import gaussian_instance
from scipy.stats import norm as ndist

def test_selected_targets(n=100,
                          p=500,
                          signal_fac=0.2,
                          s=10,
                          sigma=3.,
                          rho=0.4,
                          randomizer_scale=1.):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        if nonzero.sum() > 0:
            dispersion = None

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            estimate, observed_info_mean, _, _, _, _ = conv.selective_MLE(observed_target,
                                                                    cov_target,
                                                                    cov_target_score,
                                                                    alternatives)

            index = np.random.permutation(n)[0]
            contrast = ((X[:, nonzero])[index,:])
            target = contrast.T.dot(np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta)))
            est = contrast.T.dot(estimate)
            var_est = contrast.T.dot(observed_info_mean).dot(contrast)
            quantile = ndist.ppf(1 - 0.05)
            intervals = np.vstack([est - quantile * np.sqrt(var_est),
                                   est + quantile * np.sqrt(var_est)]).T
            pivot = ndist.cdf((est-target)/np.sqrt(var_est))

            coverage = (target > intervals[0,0]) * (target < intervals[0,1])
            return coverage, pivot

import matplotlib.pyplot as plt
from statsmodels.distributions import ECDF

def main(nsim=500):
    cover= 0.
    pivot = []

    for i in range(nsim):
        cover_, pivot_ = test_selected_targets()

        cover += cover_
        pivot.append(pivot_)

        print("iteration completed ", i)
        print("coverage so far ", cover/(i+1.))
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

main(nsim=500)
