from __future__ import print_function

import numpy as np, sys

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import numpy as np
import time
import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.tests.instance import logistic_instance, gaussian_instance

from selection.reduced_optimization.par_carved_reduced import selection_probability_carved, sel_inf_carved

#from selection.reduced_optimization.estimator import M_estimator_approx_carved
from selection.randomized.M_estimator import M_estimator, M_estimator_split
from selection.randomized.glm import pairs_bootstrap_glm, bootstrap_cov

import sys
import os

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

def generate_data_random(n, p, sigma=1., rho=0., scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.9:
            #beta_true[i] = np.random.laplace(loc=0., scale=0.1)
            beta_true[i] = np.random.normal(0., 0.1, 1)
        else:
            #beta_true[i] = np.random.laplace(loc=0., scale=3.)
            beta_true[i] = np.random.normal(0., 3.5, 1)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta * sigma, sigma

class M_estimator_approx_carved(M_estimator_split):

    def __init__(self, loss, epsilon, subsample_size, penalty, estimation):

        M_estimator_split.__init__(self,loss, epsilon, subsample_size, penalty, solve_args={'min_its':50, 'tol':1.e-10})
        self.estimation = estimation

    def solve_approx(self):

        self.solve()

        self.nactive = self._overall.sum()
        X, _ = self.loss.data
        n, p = X.shape
        self.p = p
        self.target_observed = self.observed_score_state[:self.nactive]

        self.feasible_point = np.concatenate([self.observed_score_state, np.fabs(self.observed_opt_state[:self.nactive]),
                                              self.observed_opt_state[self.nactive:]], axis = 0)

        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)

        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)

        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)

        #print("True or false", np.all(lagrange[0]-np.fabs(self.feasible_point[p+self.nactive:]))>0)
        #print("True or false", np.all(self.feasible_point[p:][:self.nactive]) > 0)

        self.inactive_lagrange = lagrange[~self._overall]

        self.bootstrap_score, self.randomization_cov = self.setup_sampler()

        if self.estimation == 'parametric':
            score_cov = np.zeros((p,p))
            inv_X_active = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
            projection_X_active = X[:,self._overall].dot(np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))).dot(X[:,self._overall].T)
            score_cov[:self.nactive, :self.nactive] = inv_X_active
            score_cov[self.nactive:, self.nactive:] = X[:,~self._overall].T.dot(np.identity(n)- projection_X_active).dot(X[:,~self._overall])

        elif self.estimation == 'bootstrap':
            score_cov = bootstrap_cov(lambda: np.random.choice(n, size=(n,), replace=True), self.bootstrap_score)

        self.score_cov = score_cov
        self.score_cov_inv = np.linalg.inv(self.score_cov)

def carved_lasso_trial(X,
                       y,
                       beta,
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
        subsample_size = int(0.50 * total_size)
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

        deno = float(beta.T.dot(X.T.dot(X)).dot(beta))

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

            adjusted_estimate = np.zeros(p)
            adjusted_estimate[active] = selective_mean
            unadjusted_estimate = np.zeros(p)
            unadjusted_estimate[active] = post_mean
            split_estimate = np.zeros(p)
            split_estimate[active] = post_mean_split

            risk_ad = (np.power(adjusted_estimate - beta, 2.).sum())/deno
            risk_unad = (np.power(unadjusted_estimate - beta, 2.).sum())/deno
            risk_split = (np.power(split_estimate - beta, 2.).sum())/deno

            ts = float(true_set.shape[0])
            print("inferential powers", power_ad / ts, power_unad / ts, power_split / ts)
            break

    if True:
        return np.vstack([sel_cov, naive_cov, split_cov, ad_len, unad_len, split_len, risk_ad, risk_unad, risk_split,
                          power_ad/ts, power_unad/ts, power_split/ts])

if __name__ == "__main__":

    ### set parameters
    n = 500
    p = 100
    s = 0
    snr = 0.

    niter = 10
    nf = 0

    ad_cov = 0.
    unad_cov = 0.
    split_cov = 0.
    Lee_cov = 0.

    ad_len = 0.
    unad_len = 0.
    split_len = 0.
    Lee_len = 0.
    Lee_inf = 0.

    no_sel = 0
    ad_risk = 0.
    unad_risk = 0.
    split_risk = 0.
    Lee_risk = 0.

    ad_power = 0.
    unad_power = 0.
    split_power = 0.
    Lee_power = 0.

    for i in range(niter):
        np.random.seed(i+30)
        X, y, beta, sigma = generate_data_random(n=n, p=p)
        print("snr", beta.T.dot(X.T.dot(X)).dot(beta)/n)

        true_mean = X.dot(beta)
        X_scaled = np.sqrt(n) * X
        deno = float(beta.T.dot(X.T.dot(X)).dot(beta))

        lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        #glm_LASSO_1se, _, lam_min, lam_1se = glmnet_lasso(X_scaled, y)

        true_signals = np.zeros(p, np.bool)
        delta = 1.e-1
        true_signals[np.abs(beta) > delta] = 1
        true_set = np.asarray([u for u in range(p) if true_signals[u]])

        glm_LASSO = glmnet_lasso(X_scaled, y, lam/np.sqrt(n))
        active_LASSO = (glm_LASSO != 0)
        active_set_LASSO = np.asarray([t for t in range(p) if active_LASSO[t]])
        active_bool_LASSO = np.zeros(active_LASSO.sum(), np.bool)
        for x in range(active_LASSO.sum()):
            active_bool_LASSO[x] = (np.in1d(active_set_LASSO[x], true_set).sum() > 0)
        #print("compare scales", lam, np.sqrt(n)*lam_1se, np.sqrt(n)*lam_min)

        Lee_intervals, Lee_pval = selInf_R(X_scaled, y, glm_LASSO, np.sqrt(n) * lam, sigma, Type=0, alpha=0.1)
        true_target = np.linalg.pinv(X_scaled[:, active_LASSO]).dot(true_mean)
        #print("true target, Lee_intervals", true_target, Lee_intervals, glm_LASSO)

        if (Lee_pval.shape[0] == true_target.shape[0]):
            lasso = carved_lasso_trial(X,
                                       y,
                                       beta,
                                       sigma,
                                       lam,
                                       true_set)

            ad_cov += lasso[0, 0]
            unad_cov += lasso[1, 0]
            split_cov += lasso[2, 0]
            Lee_cov += np.mean((true_target > Lee_intervals[:, 0]) * (true_target < Lee_intervals[:, 1]))

            ad_len += lasso[3, 0]
            unad_len += lasso[4, 0]
            split_len += lasso[5, 0]
            inf_entries = np.isinf(Lee_intervals[:, 1] - Lee_intervals[:, 0])
            Lee_inf += np.mean(inf_entries)
            if inf_entries.sum() == active_LASSO.sum():
                Lee_len += 0.
            else:
                Lee_len += np.sqrt(n) * np.mean((Lee_intervals[:, 1] - Lee_intervals[:, 0])[~inf_entries])

            ad_risk += lasso[6, 0]
            unad_risk += lasso[7, 0]
            split_risk += lasso[8, 0]
            Lee_risk += np.mean(np.power(np.sqrt(n) * glm_LASSO - beta, 2.).sum()) / deno

            ad_power += lasso[9, 0]
            unad_power += lasso[10, 0]
            split_power += lasso[11, 0]
            Lee_power += ((active_bool_LASSO) * (
            np.logical_or((0. < Lee_intervals[:, 0]), (0. > Lee_intervals[:, 1])))).sum() \
                         / float(true_set.shape[0])

            print("\n")
            print("iteration completed", i + 1-nf)
            print("\n")
            print("adjusted and unadjusted coverage so far ", ad_cov / float(i + 1 - nf), unad_cov / float(i + 1 -nf),
                  split_cov / float(i + 1-nf), Lee_cov / float(i + 1-nf))
            print("adjusted and unadjusted lengths so far ", ad_len / float(i + 1-nf), unad_len / float(i + 1-nf),
                  split_len / float(i + 1-nf), Lee_len / float(i + 1-nf))
            print("adjusted and unadjusted risks so far ", ad_risk / float(i + 1-nf), unad_risk / float(i + 1-nf),
                  split_risk / float(i + 1-nf), Lee_risk / float(i + 1-nf))
            print("adjusted and unadjusted powers so far ", ad_power / float(i + 1-nf), unad_power / float(i + 1-nf),
                  split_power / float(i + 1-nf), Lee_power / float(i + 1-nf))
            print("proportion of Lee intervals that are infty ", Lee_inf / float(i + 1-nf))

        else:
            nf += 1

