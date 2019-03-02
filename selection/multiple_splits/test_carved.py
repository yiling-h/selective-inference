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

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats

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


def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)

                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)

    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    estimate_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se')))
    return estimate, estimate_1se, estimate_min, lam_min, lam_1se

def test_carved_lasso(n=500, p=100, nval=500, rho=0.20, s=5, beta_type=1, snr=0.20, subsample_frac=0.80):

    while True:
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()
        print("sigma ", sigma)
        if n<p:
            dispersion = None
            sigma_ = np.std(y)
            #sigma_ = sigma
        else:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)

        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        randomization_cov = ((sigma_**2)*((1.-subsample_frac)/subsample_frac))* Sigma

        carved_lasso_sol = carved_lasso.gaussian(X,
                                                 y,
                                                 noise_variance= sigma_ ** 2.,
                                                 rand_covariance = "None",
                                                 randomization_cov = randomization_cov,
                                                 feature_weights=lam_theory* np.ones(p),
                                                 subsample_frac= subsample_frac)

        signs = carved_lasso_sol.fit()
        nonzero = signs != 0
        print("solution", nonzero.sum())

        if nonzero.sum() > 0:

            if n>p:
                target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = selected_targets(carved_lasso_sol.loglike,
                                                  carved_lasso_sol._W,
                                                  nonzero,
                                                  dispersion=dispersion)

                MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = carved_lasso_sol.selective_MLE(observed_target,
                                                                                                                     cov_target,
                                                                                                                     cov_target_score,
                                                                                                                     alternatives)
                print("MLE intervals ", MLE_intervals, target_randomized)

            else:
                target_randomized = beta[nonzero]
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = debiased_targets(carved_lasso_sol.loglike,
                                                  carved_lasso_sol._W,
                                                  nonzero,
                                                  penalty=carved_lasso_sol.penalty,
                                                  dispersion=dispersion)

                MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = carved_lasso_sol.selective_MLE(observed_target,
                                                                                                                     cov_target,
                                                                                                                     cov_target_score,
                                                                                                                     alternatives)
                print("MLE intervals ", MLE_intervals, target_randomized)

            return (MLE_estimate-target_randomized), np.mean((target_randomized > MLE_intervals[:, 0]) * (target_randomized < MLE_intervals[:, 1]))

def main(nsim=500):
    cover = 0.
    nn = []
    bias = 0.

    for i in range(nsim):
        nn_, cover_ = test_carved_lasso()
        cover += cover_
        bias += np.mean(nn_)
        nn.extend(nn_)
        print("completed ", i, cover / float(i + 1), bias / float(i + 1))

    # sns.distplot(np.asarray(nn))
    # plt.show()
    stats.probplot(np.asarray(nn), dist="norm", plot=pylab)
    pylab.show()

main(nsim=500)

