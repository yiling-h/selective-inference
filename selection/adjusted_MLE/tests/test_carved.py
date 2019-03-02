import numpy as np, sys, time

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, carved_lasso, full_targets, selected_targets, debiased_targets
from selection.algorithms.lasso import lasso_full
import regreg.api as rr
from selection.algorithms.debiased_lasso import debiasing_matrix
from selection.tests.instance import gaussian_instance, nonnormal_instance
from statsmodels.distributions.empirical_distribution import ECDF

def test_carved(n= 500,
                p= 100,
                signal_fac= 1.,
                s= 5,
                sigma= 1.,
                rho= 0.40,
                split_proportion= 0.50):

    inst = nonnormal_instance
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, y, beta = inst(n=n,
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

        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
        print("sigma estimated and true ", sigma, sigma_)
        randomization_cov = ((sigma_ ** 2) * ((1. - split_proportion) / split_proportion)) * sigmaX
        lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

        randomized_lasso = carved_lasso.gaussian(X,
                                                 y,
                                                 noise_variance=sigma_ ** 2.,
                                                 rand_covariance="True",
                                                 randomization_cov= randomization_cov/float(n),
                                                 feature_weights= np.ones(X.shape[1]) * lam_theory,
                                                 subsample_frac= split_proportion)

        signs = randomized_lasso.fit()
        nonzero = signs != 0
        if nonzero.sum() > 0:
            target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(randomized_lasso.loglike,
                                              randomized_lasso._W,
                                              nonzero,
                                              dispersion=dispersion)

            MLE_estimate, observed_info_mean, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                                 cov_target,
                                                                                                                 cov_target_score,
                                                                                                                 alternatives)

            print("inetrvals ", MLE_intervals, target_randomized)
            pivot_MLE = np.true_divide(MLE_estimate - target_randomized, np.sqrt(np.diag(observed_info_mean)))
            cov_MLE = np.mean((target_randomized >  MLE_intervals[:, 0]) * (target_randomized < MLE_intervals[:, 1]))

            return pivot_MLE, cov_MLE

def test_plot_pivot(ndraw=100):
    import matplotlib.pyplot as plt

    _pivot_MLE = []
    _cov_MLE = 0.
    for i in range(ndraw):
        pivot_MLE, cov_MLE = test_carved(n=500,
                                         p=100,
                                         signal_fac=1.8,
                                         s=5,
                                         sigma=3.,
                                         rho=0.75,
                                         split_proportion=0.50)
        _cov_MLE += cov_MLE
        for j in range(pivot_MLE.shape[0]):
            _pivot_MLE.append(pivot_MLE[j])
        print("iteration compeleted ", i)

    print("coverage of MLE", _cov_MLE/ndraw)

    # plt.clf()
    # ecdf_MLE = ECDF(ndist.cdf(np.asarray(_pivot_MLE)))
    # grid = np.linspace(0, 1, 101)
    # plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    # plt.plot(grid, grid, 'k--')
    # plt.show()

test_plot_pivot(ndraw=100)

