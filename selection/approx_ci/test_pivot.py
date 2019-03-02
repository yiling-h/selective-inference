from __future__ import division, print_function

import numpy as np
import nose.tools as nt, functools

import regreg.api as rr

from selection.randomized.lasso import lasso, carved_lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance
from selection.tests.flags import SET_SEED
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from selection.algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from selection.randomized.randomization import randomization
from selection.tests.decorators import rpy_test_safe
from selection.randomized.query import approx_density, approx_reference

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats
from selection.tests.instance import gaussian_instance

def test_approx_pivot(n= 500,
                      p= 100,
                      signal_fac= 1.,
                      s= 5,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.):

    inst = gaussian_instance
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

        n, p = X.shape

        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
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

        grid_num = 1501
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        for m in range(nonzero.sum()):
            observed_target_uni = (observed_target[m]).reshape((1,))
            cov_target_uni = (np.diag(cov_target)[m]).reshape((1,1))
            cov_target_score_uni = cov_target_score[m,:].reshape((1, p))
            mean_parameter = beta_target[m]
            grid = np.linspace(observed_target_uni - 15., observed_target_uni + 15., num=grid_num)
            grid_indx_obs = np.argmin(np.abs(grid - observed_target_uni))

            approx_log_ref= approx_reference(grid,
                                             observed_target_uni,
                                             cov_target_uni,
                                             cov_target_score_uni,
                                             conv.observed_opt_state,
                                             conv.cond_mean,
                                             conv.cond_cov,
                                             conv.logdens_linear,
                                             conv.A_scaling,
                                             conv.b_scaling,
                                             solve_args={'tol': 1.e-15})

            area_cum = approx_density(grid,
                                      mean_parameter,
                                      cov_target_uni,
                                      approx_log_ref)

            pivot = area_cum[grid_indx_obs]

            return pivot




