from __future__ import print_function
import numpy as np
import time
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.tests.instance import logistic_instance, gaussian_instance

from selection.reduced_optimization.par_carved_reduced import selection_probability_carved, sel_inf_carved
from selection.reduced_optimization.estimator import M_estimator_approx_carved, M_estimator_approx

import sys
import os

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    return estimate

def selInf_R(X, y, beta, lam, sigma, Type, alpha=0.1):
    robjects.r('''
               library("selectiveInference")
               selInf = function(X, y, beta, lam, sigma, Type, alpha= 0.1){
               y = as.matrix(y)
               X = as.matrix(X)
               beta = as.matrix(beta)
               lam = as.matrix(lam)[1,1]
               sigma = as.matrix(sigma)[1,1]
               Type = as.matrix(Type)[1,1]
               if(Type == 1){
                   type = "full"} else{
                   type = "partial"}
               inf = fixedLassoInf(x = X, y = y, beta = beta, lambda=lam, family = "gaussian",
                                   intercept=FALSE, sigma=sigma, alpha=alpha, type=type)
               return(list(ci = inf$ci, pvalue = inf$pv))}
               ''')

    inf_R = robjects.globalenv['selInf']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_beta = robjects.r.matrix(beta, nrow=p, ncol=1)
    r_lam = robjects.r.matrix(lam, nrow=1, ncol=1)
    r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)
    r_Type = robjects.r.matrix(Type, nrow=1, ncol=1)
    output = inf_R(r_X, r_y, r_beta, r_lam, r_sigma, r_Type)
    ci = np.array(output.rx2('ci'))
    pvalue = np.array(output.rx2('pvalue'))
    return ci, pvalue


def carved_lasso_trial(X,
                       y,
                       beta,
                       true_mean,
                       sigma,
                       lam,
                       true_set,
                       estimation='parametric'):

    while True:
        n, p = X.shape
        loss = rr.glm.gaussian(X, y)
        epsilon = 1. / np.sqrt(n)

        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

        total_size = loss.saturated_loss.shape[0]
        # fix subsample size for running LASSO
        subsample_size = int(0.5 * total_size)
        inference_size = total_size-subsample_size

        M_est = M_estimator_approx_carved(loss, epsilon, subsample_size, penalty, estimation)
        sel_indx = M_est.sel_indx
        X_inf = X[~sel_indx, :]
        y_inf = y[~sel_indx]
        M_est.solve_approx()

        active = M_est._overall
        nactive = M_est.nactive
        active_set = np.asarray([t for t in range(p) if active[t]])
        active_bool = np.zeros(nactive, np.bool)
        for x in range(nactive):
            active_bool[x] = (np.in1d(active_set[x], true_set).sum() > 0)

        if nactive >= 1:
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
            coverage_split = (true_val_split > adjusted_intervals_split[0, :]) * (true_val_split < adjusted_intervals_split[1, :])
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

            print("inferential powers", power_ad/5., power_unad/5., power_split/5.)
            break

    if True:
        return np.vstack([sel_cov, naive_cov, split_cov, ad_len, unad_len, split_len, risk_ad, risk_unad, risk_split])

# if __name__ == "__main__":
#     ### set parameters
#     n = 1000
#     p = 100
#     s = 5
#     snr = 5.
#     rho = 0.
#
#     niter = 10
#     ad_cov = 0.
#     unad_cov = 0.
#     split_cov = 0.
#     ad_len = 0.
#     unad_len = 0.
#     split_len = 0.
#     no_sel = 0
#     ad_risk = 0.
#     unad_risk = 0.
#     split_risk = 0.
#
#     for i in range(niter):
#
#          ### GENERATE X, Y BASED ON SEED
#          np.random.seed(i+30)  # ensures different y
#          X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=rho, snr=snr)
#          true_mean = X.dot(beta)
#
#          idx = np.arange(p)
#          sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
#          print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))
#
#          lam = 0.8 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
#
#          true_signals = np.zeros(p, np.bool)
#          true_signals[beta != 0] = 1
#          true_set = np.asarray([u for u in range(p) if true_signals[u]])
#          ### RUN LASSO AND TEST
#          lasso = carved_lasso_trial(X,
#                                     y,
#                                     beta,
#                                     true_mean,
#                                     sigma,
#                                     lam,
#                                     true_set)
#
#          ad_cov += lasso[0, 0]
#          unad_cov += lasso[1, 0]
#          split_cov += lasso[2, 0]
#
#          ad_len += lasso[3, 0]
#          unad_len += lasso[4, 0]
#          split_len += lasso[5, 0]
#
#          ad_risk += lasso[6, 0]
#          unad_risk += lasso[7, 0]
#          split_risk += lasso[8, 0]
#          print("\n")
#          print("iteration completed", i+1)
#          print("\n")
#          print("adjusted and unadjusted coverage so far ", ad_cov/float(i+1), unad_cov/float(i+1), split_cov/float(i+1))
#          print("adjusted and unadjusted lengths so far ", ad_len/float(i+1), unad_len/float(i+1), split_len/float(i+1))
#          print("adjusted and unadjusted risks so far ", ad_risk/float(i+1), unad_risk/float(i+1), split_risk/float(i+1))

def unadjusted_inference(X,
                         y,
                         beta,
                         sigma,
                         lam,
                         true_set,
                         randomizer='gaussian',
                         estimation='parametric'):

    while True:
        from selection.api import randomization
        n, p = X.shape

        loss = rr.glm.gaussian(X, y)
        epsilon = 1. / np.sqrt(n)

        W = np.ones(p) * lam
        penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)

        randomization = randomization.isotropic_gaussian((p,), scale=.1)

        M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer , estimation)
        M_est.solve_approx()

        active = M_est._overall
        active_set = np.asarray([t for t in range(p) if active[t]])
        nactive = M_est.nactive
        active_bool = np.zeros(nactive, np.bool)
        for x in range(nactive):
            active_bool[x] = (np.in1d(active_set[x], true_set).sum() > 0)

        deno = float(beta.T.dot(X.T.dot(X)).dot(beta))

        if nactive >= 1:
            true_mean = X.dot(beta)
            prior_variance = 1000.
            noise_variance = sigma ** 2
            projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
            true_val = projection_active.T.dot(true_mean)

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

            naive_cov = np.mean(coverage_unad)
            unad_len = np.mean(unad_length)
            unadjusted_estimate = np.zeros(p)
            unadjusted_estimate[active] = post_mean
            risk_unad = (np.power(unadjusted_estimate - beta, 2.).sum()) / deno
            break

    if True:
        return np.vstack([naive_cov, unad_len, risk_unad])


if __name__ == "__main__":

    ### set parameters
    n = 500
    p = 100
    s = 5
    snr = 0.
    rho = 0.2
    SNR = 0.

    niter = 50
    nf = 0

    unad_cov = 0.
    unad_len = 0.
    unad_risk = 0.

    Lee_cov = 0.
    Lee_inf = 0.
    Lee_len = 0.

    for i in range(niter):
        np.random.seed(i)
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=rho, snr=snr)
        print("beta", beta[beta!=0])
        print("snr", beta.T.dot(X.T.dot(X)).dot(beta)/n)
        SNR += beta.T.dot(X.T.dot(X)).dot(beta)/n
        print("snr so far", SNR/float(i+1))

        X_scaled = np.sqrt(n) * X

        true_mean = X.dot(beta)
        X_scaled = np.sqrt(n) * X
        deno = float(beta.T.dot(X.T.dot(X)).dot(beta))
        prior_variance = 1000.
        noise_variance = sigma ** 2

        lam = 0.75 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

        true_signals = np.zeros(p, np.bool)
        delta = 1.e-1
        true_signals[np.abs(beta) > delta] = 1
        true_set = np.asarray([u for u in range(p) if true_signals[u]])

        glm_LASSO = glmnet_lasso(X_scaled, y, lam / np.sqrt(n))
        active_LASSO = (glm_LASSO != 0)
        if active_LASSO.sum()>0:
            Lee_intervals, Lee_pval = selInf_R(X_scaled, y, glm_LASSO, np.sqrt(n) * lam, sigma, Type=0, alpha=0.1)
            true_target = np.linalg.pinv(X_scaled[:, active_LASSO]).dot(true_mean)

            lasso = unadjusted_inference(X,
                                         y,
                                         beta,
                                         sigma,
                                         lam,
                                         true_set)

            unad_cov += lasso[0, 0]
            unad_len += lasso[1, 0]
            unad_risk += lasso[2, 0]

            print("\n")
            print("iteration completed", i + 1 - nf)
            print("\n")
            print("unadjusted coverage so far ", unad_cov / float(i + 1 - nf))
            print("unadjusted lengths so far ", unad_len / float(i + 1 - nf))
            print("unadjusted risks so far ", unad_risk / float(i + 1 - nf))

            if (Lee_pval.shape[0] == true_target.shape[0]):
                Lee_cov += np.mean((true_target > Lee_intervals[:, 0]) * (true_target < Lee_intervals[:, 1]))
                inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
                Lee_inf += np.mean(inf_entries)
                if inf_entries.sum() == active_LASSO.sum():
                    Lee_len += 0.
                else:
                    Lee_len += np.sqrt(n) * np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])

                print("\n")
                print("Lee coverage so far ", Lee_cov / float(i + 1 - nf))
                print("Lee lengths so far ", Lee_len / float(i + 1 - nf), Lee_inf / float(i + 1 - nf))


            else:
                nf += 1
        else:
            nf += 1




