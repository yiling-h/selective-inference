from __future__ import division, print_function

import numpy as np
from selection.randomized.lasso import lasso, carved_lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance, nonnormal_instance, mixed_normal_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


import matplotlib.pyplot as plt
from selection.approx_ci.approx_reference import approx_reference, approx_density

def test_approx_pivot(n= 500,
                      p= 100,
                      signal_fac= 1.,
                      s= 5,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.):

    #inst = gaussian_instance
    inst = nonnormal_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
        #sigma_ = np.std(y)
        print("sigma estimated and true ", sigma, sigma_)

        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = lasso.gaussian(X,
                              y,
                              W,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid_num = 361
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        pivot = []
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1,1))
            cov_target_score_uni = cov_target_score[m,:].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(- 18., 18., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))
            print("check grid position ", observed_target_uni, grid_indx_obs)

            approx_log_ref= approx_reference(grid,
                                             observed_target_uni,
                                             cov_target_uni,
                                             cov_target_score_uni,
                                             conv.observed_opt_state,
                                             conv.cond_mean,
                                             conv.cond_cov,
                                             conv.logdens_linear,
                                             conv.A_scaling,
                                             conv.b_scaling)

            area_cum = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref)

            pivot.append(1. - area_cum[grid_indx_obs])
            print("variable completed ", m+1)
        return pivot

from statsmodels.distributions.empirical_distribution import ECDF


def test_approx_pivot_carved(n= 100,
                             p= 50,
                             signal_fac= 1.,
                             s= 5,
                             sigma= 1.,
                             rho= 0.40,
                             split_proportion=0.50):

    inst = nonnormal_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

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

        conv = carved_lasso.gaussian(X,
                                     y,
                                     noise_variance=sigma_ ** 2.,
                                     rand_covariance="None",
                                     randomization_cov=randomization_cov / float(n),
                                     feature_weights=np.ones(X.shape[1]) * lam_theory,
                                     subsample_frac=split_proportion)

        signs = conv.fit()
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        grid_num = 361
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        pivot = []
        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1, 1))
            cov_target_score_uni = cov_target_score[m, :].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(- 18., 18., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))

            approx_log_ref = approx_reference(grid,
                                              observed_target_uni,
                                              cov_target_uni,
                                              cov_target_score_uni,
                                              conv.observed_opt_state,
                                              conv.cond_mean,
                                              conv.cond_cov,
                                              conv.logdens_linear,
                                              conv.A_scaling,
                                              conv.b_scaling)

            area_cum = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref)

            pivot.append(1. - area_cum[grid_indx_obs])
            print("variable completed ", m + 1)
        return pivot

def main(nsim=150):
    _pivot=[]
    for i in range(nsim):
        _pivot.extend(test_approx_pivot_carved())
        print("iteration completed ", i)
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(_pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

main()


