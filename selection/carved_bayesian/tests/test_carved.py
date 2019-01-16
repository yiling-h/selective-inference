from __future__ import print_function
import numpy as np, sys, os
import time
import regreg.api as rr

from selection.carved_bayesian.carved_inference import (sel_inf_carved,
                                                        smooth_cube_barrier)

from selection.carved_bayesian.carved_lasso import M_estimator_approx_carved

def generate_data_random(n, p, sigma=1., rho=0.2, scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.9:
            beta_true[i] = np.random.normal(0., 0.1, 1)
        else:
            beta_true[i] = np.random.normal(0., 3., 1)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta * sigma, sigma

def carved_lasso_trial(X,
                       y,
                       beta,
                       sigma,
                       lam,
                       true_set,
                       split_proportion = 0.75,
                       estimation='parametric'):
    while True:
        n, p = X.shape

        loss = rr.glm.gaussian(X, y)
        epsilon = 1. / np.sqrt(n)

        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

        total_size = loss.saturated_loss.shape[0]
        subsample_size = int(split_proportion * total_size)
        inference_size = total_size - subsample_size

        M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, estimation)
        sel_indx = M_est.sel_indx
        X_inf = X[~sel_indx, :]
        y_inf = y[~sel_indx]
        M_est.solve_approx()

        active = M_est._overall
        active_set = np.asarray([t for t in range(p) if active[t]])
        nactive = M_est.nactive
        active_bool = np.zeros(nactive, np.bool)
        for x in range(nactive):
            active_bool[x] = (np.in1d(active_set[x], true_set).sum() > 0)

        if nactive >= 1:
            true_mean = X.dot(beta)
            prior_variance = 1000.
            noise_variance = sigma ** 2
            projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
            true_val = projection_active.T.dot(true_mean)
            projection_active_split = X_inf[:, active].dot(np.linalg.inv(X_inf[:, active].T.dot(X_inf[:, active])))
            true_val_split = projection_active_split.T.dot(X_inf.dot(beta))

            M_1 = prior_variance * (X.dot(X.T)) + noise_variance * np.identity(n)
            M_2 = prior_variance * ((X.dot(X.T)).dot(projection_active))
            M_3 = prior_variance * (projection_active.T.dot(X.dot(X.T)).dot(projection_active))
            post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)
            post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)
            unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                              post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])
            coverage_unad = (true_val > unadjusted_intervals[0, :]) * (true_val < unadjusted_intervals[1, :])
            unad_length = unadjusted_intervals[1, :] - unadjusted_intervals[0, :]
            power_unad = ((active_bool) * (np.logical_or((0. < unadjusted_intervals[0, :]),
                                                         (0. > unadjusted_intervals[1, :])))).sum()

            M_1_split = prior_variance * (X_inf.dot(X_inf.T)) + noise_variance * np.identity(int(inference_size))
            M_2_split = prior_variance * ((X_inf.dot(X_inf.T)).dot(projection_active_split))
            M_3_split = prior_variance * (
                projection_active_split.T.dot(X_inf.dot(X_inf.T)).dot(projection_active_split))
            post_mean_split = M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(y_inf)
            post_var_split = M_3_split - M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(M_2_split)
            adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(post_var_split.diagonal())),
                                                  post_mean_split + 1.65 * (np.sqrt(post_var_split.diagonal()))])
            coverage_split = (true_val_split > adjusted_intervals_split[0, :]) * (
            true_val_split < adjusted_intervals_split[1, :])
            split_length = adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :]
            power_split = ((active_bool) * (np.logical_or((0. < adjusted_intervals_split[0, :]),
                                                          (0. > adjusted_intervals_split[1, :])))).sum()

            grad_lasso = sel_inf_carved(M_est, prior_variance)
            samples = grad_lasso.posterior_samples()
            adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
            coverage_ad = (true_val > adjusted_intervals[0, :]) * (true_val < adjusted_intervals[1, :])
            ad_length = adjusted_intervals[1, :] - adjusted_intervals[0, :]
            selective_mean = np.mean(samples, axis=0)
            power_ad = ((active_bool) * (np.logical_or((0. < adjusted_intervals[0, :]),
                                                       (0. > adjusted_intervals[1, :])))).sum()

            sel_cov = np.mean(coverage_ad)
            naive_cov = np.mean(coverage_unad)
            split_cov = np.mean(coverage_split)
            ad_len = np.mean(ad_length)
            unad_len = np.mean(unad_length)
            split_len = np.mean(split_length)
            risk_ad = np.mean(np.power(selective_mean - true_val, 2.))
            risk_unad = np.mean(np.power(post_mean - true_val, 2.))
            risk_split = np.mean(np.power(post_mean_split - true_val_split, 2.))

            ts = float(true_set.shape[0])
            break

    if True:
        return np.vstack([sel_cov, naive_cov, split_cov, ad_len, unad_len, split_len, risk_ad, risk_unad, risk_split,
                          power_ad/ts, power_unad/ts, power_split/ts])

if __name__ == "__main__":

    ### set parameters
    n = 500
    p = 100

    niter = 10
    results = np.zeros(12)

    for i in range(niter):
        np.random.seed(i)
        X, y, beta, sigma = generate_data_random(n=n, p=p)
        lam = 0.8 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

        true_signals = np.zeros(p, np.bool)
        delta = 1.e-1
        true_signals[np.abs(beta) > delta] = 1
        true_set = np.asarray([u for u in range(p) if true_signals[u]])

        lasso = carved_lasso_trial(X,
                                   y,
                                   beta,
                                   sigma,
                                   lam,
                                   true_set,
                                   split_proportion=0.50)

        results += lasso

        print("\n")
        print("iteration completed", i + 1)
        print("\n")
        print("results so far", results/float(i+1))
