from __future__ import division, print_function

import numpy as np
from selection.randomized.lasso import lasso, carved_lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance, nonnormal_instance, mixed_normal_instance
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from selection.approx_ci.approx_reference import approx_reference, approx_density, approx_ci

def test_approx_ci_carved(n= 200,
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

        if nonzero.sum()>0:
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            grid_num = 361
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            coverage = 0.
            length = 0.

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

                param_grid = np.linspace(-12., 12., num=241)
                approx_lci, approx_uci = approx_ci(param_grid,
                                                   grid,
                                                   cov_target_uni,
                                                   approx_log_ref,
                                                   grid_indx_obs)

                print("variable completed ", m + 1)
                coverage += (approx_lci < mean_parameter) * (approx_uci > mean_parameter)
                length += (approx_uci - approx_lci)
            return coverage / float(nonzero.sum()), length / float(nonzero.sum())


def main(nsim=150):
    _coverage = 0.
    _length = 0.
    for i in range(nsim):
        cov, len =  test_approx_ci_carved()
        _coverage += cov
        _length += len
        print("iteration completed ", i)
        print("coverage and lengths so far ", _coverage/(i+1.), _length/(i+1.))

main()



