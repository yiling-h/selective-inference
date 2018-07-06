from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.tests.instance import logistic_instance, gaussian_instance

from selection.reduced_optimization.par_carved_reduced import selection_probability_carved, sel_inf_carved
from selection.reduced_optimization.estimator import M_estimator_approx_carved

import sys
import os

def carved_lasso_trial(X,
                       y,
                       beta,
                       true_mean,
                       sigma,
                       lam,
                       estimation='parametric'):

    n, p = X.shape
    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

    total_size = loss.saturated_loss.shape[0]
    #fix subsample size for running LASSO
    subsample_size = int(0.75 * total_size)

    M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, estimation)
    sel_indx = M_est.sel_indx
    X_inf = X[~sel_indx, :]
    y_inf = y[~sel_indx]
    M_est.solve_approx()

    active = M_est._overall
    nactive = M_est.nactive

    if nactive >= 1:
        prior_variance = 1000.
        noise_variance = sigma ** 2
        projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
        M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        print("observed data", post_mean)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])
        grad_lasso = sel_inf_carved(M_est, prior_variance)
        samples = grad_lasso.posterior_samples()
        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
        selective_mean = np.mean(samples, axis=0)

        coverage_ad = np.zeros(nactive)
        coverage_unad = np.zeros(nactive)
        coverage_split = np.zeros(nactive)
        ad_length = np.zeros(nactive)
        unad_length = np.zeros(nactive)
        ad_split = np.zeros(nactive)

        true_val = projection_active.T.dot(true_mean)
        for l in range(nactive):
            if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                coverage_ad[l] += 1
            ad_length[l] = adjusted_intervals[1, l] - adjusted_intervals[0, l]
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[l] += 1
            unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]

        projection_active_split = X_inf[:, active].dot(np.linalg.inv(X_inf[:, active].T.dot(X_inf[:, active])))
        M_1_split = prior_variance * (X_inf.dot(X_inf.T)) + noise_variance * np.identity(int(n / 2.)) #needs change
        M_2_split = prior_variance * ((X_inf.dot(X_inf.T)).dot(projection_active_split))
        M_3_split = prior_variance * (projection_active.T.dot(X_inf.dot(X_inf.T)).dot(projection_active_split))
        post_mean_split = M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(y_inf)
        post_var_split = M_3_split - M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(M_2_split)

        adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(post_var_split.diagonal())),
                                              post_mean_split + 1.65 * (np.sqrt(post_var_split.diagonal()))])
        split_length = adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :]
        coverage_split = (true_val > adjusted_intervals_split[0, :]) * (true_val < adjusted_intervals_split[1, :])

        sel_cov = coverage_ad.sum() / nactive
        naive_cov = coverage_unad.sum() / nactive
        split_cov = coverage_split.sum()/ nactive
        ad_len = np.mean(ad_length)
        unad_len = np.mean(unad_length)
        split_len = np.mean(split_length)
        risk_ad = np.power(selective_mean - true_val, 2.).sum() / nactive
        risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive
        risk_split = np.power(post_mean_split - true_val, 2.).sum() / nactive

        return np.vstack([sel_cov, naive_cov, split_cov, ad_len, unad_len, split_len, risk_ad, risk_unad, risk_split])

    else:
        return np.vstack([0., 0., 0., 0., 0., 0., 0., 0., 0.])
