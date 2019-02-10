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
from selection.randomized.query import selective_MLE_grid

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats

from selection.tests.instance import gaussian_instance

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
    sim_xy = sim.xy
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def cdf_mle(param, MLE_estimate_vec, MLE_var_vec):

    approx_density = np.exp(-np.true_divide((MLE_estimate_vec - param) ** 2, 2 * MLE_var_vec))
    normalized_mle_density = approx_density / (approx_density.sum())

    return np.cumsum(normalized_mle_density)


def test_saddle(n=500, p=1000, nval=500, rho=0.20, s=30, beta_type=1, snr=0.55, subsample_frac=0.80):
    while True:
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        sigma_ = sigma
        dispersion = sigma_**2
        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        randomization_cov = ((sigma_ ** 2) * ((1. - subsample_frac) / subsample_frac)) * Sigma

        carved_lasso_sol = carved_lasso.gaussian(X,
                                                 y,
                                                 noise_variance=sigma_ ** 2.,
                                                 rand_covariance="True",
                                                 randomization_cov=randomization_cov,
                                                 feature_weights=lam_theory * np.ones(p),
                                                 subsample_frac=subsample_frac)

        signs = carved_lasso_sol.fit()
        nonzero = signs != 0
        nactive = nonzero.sum()

        if nactive > 0:
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(carved_lasso_sol.loglike,
                                              carved_lasso_sol._W,
                                              nonzero,
                                              dispersion=dispersion)

            obs_MLE_estimate, _, _, _, MLE_intervals_normal, _ = carved_lasso_sol.selective_MLE(observed_target,
                                                                                                cov_target,
                                                                                                cov_target_score,
                                                                                                alternatives)

            grid_num = 1501
            MLE_estimate_vec = np.zeros((nactive, grid_num))
            MLE_var_vec = np.zeros((nactive, grid_num))
            saddle_intervals = np.zeros((nactive, 2))
            target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            for m in range(nactive):
                indx = m
                grid = np.linspace(observed_target[indx] - 15., observed_target[indx] + 15., num=grid_num)
                cond_mean_carved, cond_cov_carved, logdens_linear_carved = carved_lasso_sol.cond_mean, carved_lasso_sol.cond_cov, carved_lasso_sol.logdens_linear

                for k in range(grid.shape[0]):
                    MLE_estimate_vec[m, k], MLE_var_vec[m, k], _, _, _, _ = selective_MLE_grid(np.array([grid[k]]),
                                                                                              observed_target[indx].reshape((1,)),
                                                                                              cov_target[indx, indx].reshape((1, 1)),
                                                                                              cov_target_score[indx,:].reshape((1, p)),
                                                                                              carved_lasso_sol.observed_opt_state,
                                                                                              cond_mean_carved,
                                                                                              cond_cov_carved,
                                                                                              logdens_linear_carved,
                                                                                              carved_lasso_sol.A_scaling,
                                                                                              carved_lasso_sol.b_scaling)

                indx_obs = np.argmin(np.abs(MLE_estimate_vec[m,:] - obs_MLE_estimate[indx]))

                param_grid = np.linspace(-2., 2., num=1001)
                area = np.zeros(param_grid.shape[0])
                for l in range(param_grid.shape[0]):
                    area_vec = cdf_mle(param_grid[l], MLE_estimate_vec[m,:], MLE_var_vec[m,:])
                    area[l] = area_vec[indx_obs]
                region = param_grid[(area >= 0.05) & (area <= 0.95)]
                saddle_intervals[m,0] = np.nanmin(region)
                saddle_intervals[m,1] = np.nanmax(region)

                print("m completed ", m, nactive,  "\n")
                print("intervals ", np.nanmin(region), np.nanmax(region), "\n")
                print("normal intervals ", MLE_intervals_normal[indx, :], "\n")
                print("coverage so far ", np.mean((target_randomized[:m] > MLE_intervals_normal[:m, 0]) * (target_randomized[:m] < MLE_intervals_normal[:m, 1])),
                      np.mean((target_randomized[:m] > saddle_intervals[:m, 0]) * (target_randomized[:m] < saddle_intervals[:m, 1])))

            return np.mean((target_randomized > MLE_intervals_normal[:, 0]) * (
                        target_randomized < MLE_intervals_normal[:, 1])), \
                   np.mean((target_randomized > saddle_intervals[:, 0]) * (target_randomized < saddle_intervals[:, 1]))


def test_normal(n=500, p=1000, nval=500, rho=0.20, s=30, beta_type=1, snr=0.55, subsample_frac=0.80):
    while True:
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()

        sigma_ = sigma
        dispersion = sigma_**2
        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        randomization_cov = ((sigma_ ** 2) * ((1. - subsample_frac) / subsample_frac)) * Sigma

        carved_lasso_sol = carved_lasso.gaussian(X,
                                                 y,
                                                 noise_variance=sigma_ ** 2.,
                                                 rand_covariance="True",
                                                 randomization_cov=randomization_cov,
                                                 feature_weights=lam_theory * np.ones(p),
                                                 subsample_frac=subsample_frac)

        signs = carved_lasso_sol.fit()
        nonzero = signs != 0

        if nonzero.sum() > 0:
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(carved_lasso_sol.loglike,
                                              carved_lasso_sol._W,
                                              nonzero,
                                              dispersion=dispersion)

            obs_MLE_estimate, _, _, _, MLE_intervals_normal, _ = carved_lasso_sol.selective_MLE(observed_target,
                                                                                                cov_target,
                                                                                                cov_target_score,
                                                                                                alternatives)

            target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            return np.mean((target_randomized > MLE_intervals_normal[:, 0]) * (
                        target_randomized < MLE_intervals_normal[:, 1])), 0.


def main(nsim=500):
    cover_normal = 0.
    cover_saddle = 0.

    for i in range(nsim):
        ncover_, scover_= test_saddle(n=2000, p=3000, nval=2000, rho=0.20, s=30, beta_type=1, snr=0.55, subsample_frac=0.80)
        cover_normal += ncover_
        cover_saddle += scover_
        print("completed ", i, cover_normal/ float(i + 1), cover_saddle/ float(i + 1))

main(nsim=500)

