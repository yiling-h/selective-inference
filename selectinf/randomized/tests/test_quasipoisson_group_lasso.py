import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import time

import regreg.api as rr

from selectinf.randomized.group_lasso_query import (group_lasso,split_group_lasso)
from selectinf.randomized.group_lasso_query_quasi import (group_lasso_quasi, split_group_lasso_quasi)

from ...base import (selected_targets,selected_targets_quasi,full_targets_quasi)
from selectinf.randomized.tests.instance import (quasi_poisson_group_instance,
                                                 poisson_group_instance)

from ...base import restricted_estimator
import scipy.stats

def calculate_F1_score(beta_true, selection):
    p = len(beta_true)
    nonzero_true = (beta_true != 0)

    # precision & recall
    if selection.sum() > 0:
        precision = (nonzero_true * selection).sum() / selection.sum()
    else:
        precision = 0
    recall = (nonzero_true * selection).sum() / nonzero_true.sum()

    if precision + recall > 0:
        return 2 * precision * recall / (precision + recall)
    else:
        return 0

def naive_inference(X, Y, groups, beta, const,
                    n, weight_frac=1, level=0.9, nonzero_true = None):
    p = X.shape[1]
    if nonzero_true is not None:
        print("(Naive) True E used")
        nonzero = nonzero_true
        print('Naive selection', nonzero_true)
    else:
        p = X.shape[1]
        sigma_ = np.std(Y)
        # weights = dict([(i, 0.5) for i in np.unique(groups)])
        weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

        conv = const(X=X,
                     counts=Y,
                     groups=groups,
                     weights=weights,
                     useJacobian=True,
                     perturb=np.zeros(p),
                     ridge_term=0.)

        signs, _ = conv.fit()
        nonzero = signs != 0

        # print('Naive selection', conv._ordered_groups)


    # Solving the inferential target
    def solve_target_restricted():
        Y_mean = np.exp(X.dot(beta))

        loglike_Mean = rr.glm.poisson(X, counts=Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike_Mean,
                                                 nonzero)
        return _beta_unpenalized

    target = solve_target_restricted()

    """_compare_H_K = False
    if _compare_H_K:
        H_K_EE_true_quality = {}
        H_K_EE_true_quality["H_EE* hat norm"] = []
        H_K_EE_true_quality["K_EE*_hat_sub norm"] = []
        H_K_EE_true_quality["K_EE*_hat_full norm"] = []
        H_K_EE_true_quality["H_EE* - K_EE*_hat_sub norm"] = []
        H_K_EE_true_quality["H_EE* - K_EE*_hat_full norm"] = []
        H_K_EE_true_quality["K_EE*_hat_sub - K_EE*_hat_full norm"] = []
        H_K_EE_true_quality["H_EE*(sub)_hat_inv - cov_sub norm"] = []
        H_K_EE_true_quality["H_EE*(sub)_hat_inv - cov_full norm"] = []
        H_K_EE_true_quality["cov_sub - cov_full norm"] = []"""

    if nonzero.sum() > 0:
        # E: nonzero flag

        X_E = X[:, nonzero]

        loglike = rr.glm.poisson(X, counts=Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE = restricted_estimator(loglike, nonzero)
        beta_MLE_full = restricted_estimator(loglike,  # refit OLS (MLE)
                                             active=np.array([True] * p))

        W_K = np.diag((Y - np.exp(X_E @ beta_MLE)) ** 2)
        cov_score = X.T @ W_K @ X
        if nonzero_true is not None:
            solution = np.zeros(p)
            solution[nonzero] = beta_MLE
            target_spec = selected_targets_quasi(loglike,
                                                 solution,
                                                 cov_score=cov_score,
                                                 dispersion=1)
        else:
            target_spec = selected_targets_quasi(conv.loglike,
                                                 conv.observed_soln,
                                                 cov_score=cov_score,
                                                 dispersion=1)
        cov = target_spec.cov_target
        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        z_low = scipy.stats.norm.ppf((1 - level) / 2)
        z_up = scipy.stats.norm.ppf(1 - (1 - level) / 2)
        assert np.abs(np.abs(z_low) - np.abs(z_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE + z_low * sd
        intervals_up = beta_MLE + z_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        return coverage, intervals_up - intervals_low, nonzero, intervals_low, intervals_up, target

    return None, None, None, None, None, None

def randomization_inference_poisson(X, Y, n, p, beta, groups, hess=None,
                                    weight_frac=1, level=0.9, solve_only=False):
    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    def estimate_hess():
        loglike = rr.glm.poisson(X, counts=Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_full = restricted_estimator(loglike, np.array([True] * p))
        W_H = np.diag(np.exp(X @ beta_full))
        return X.T @ W_H @ X

    if hess is None:
        print("(MLE Poisson) H estimated with full model")
        hess = estimate_hess()

    sigma_ = np.std(Y)
    # weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = group_lasso.poisson(X=X,
                               counts=Y,
                               groups=groups,
                               weights=weights,
                               useJacobian=True,
                               ridge_term=0.,
                               cov_rand=hess)


    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        Y_mean = np.exp(X.dot(beta))

        loglike = rr.glm.poisson(X, counts=Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    # Return the selected variables if we only want to solve the problem
    if solve_only:
        return None, None, solve_target_restricted(), nonzero, None, None

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=1)

        result = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, nonzero, \
               intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None

def data_splitting_poisson(X, Y, n, p, const, beta, nonzero=None, subset_select=None, groups=None,
                   weight_frac=1., proportion=0.5, level=0.9):

    if (nonzero is None) or (subset_select is None):
        print("(Poisson Data Splitting) Selection done without carving")
        pi_s = proportion
        subset_select = np.zeros(n, np.bool)
        subset_select[:int(pi_s * n)] = True
        n1 = subset_select.sum()
        n2 = n - n1
        np.random.shuffle(subset_select)
        X_S = X[subset_select, :]
        Y_S = Y[subset_select]

        # Selection on the first subset of data
        p = X.shape[1]
        sigma_ = np.std(Y_S)
        # weights = dict([(i, 0.5) for i in np.unique(groups)])
        weights = dict([(i, (n1/n)*weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

        conv = const(X=X_S,
                     counts=Y_S,
                     groups=groups,
                     weights=weights,
                     useJacobian=True,
                     perturb=np.zeros(p),
                     ridge_term=0.)

        signs, _ = conv.fit()
        # print("signs",  signs)
        nonzero = signs != 0

    n1 = subset_select.sum()
    n2 = n - n1

    if nonzero.sum() > 0:
        # Solving the inferential target
        def solve_target_restricted():
            Y_mean = np.exp(X.dot(beta))

            loglike = rr.glm.poisson(X, counts=Y_mean)
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # E: nonzero flag

        X_notS_E = X_notS[:, nonzero]

        loglike = rr.glm.poisson(X_notS, counts=Y_notS)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE_notS = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(np.exp(X_notS_E @ beta_MLE_notS))

        f_info = X_notS_E.T @ W @ X_notS_E * (n / n2)
        cov = np.linalg.inv(f_info)

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        z_low = scipy.stats.norm.ppf((1 - level) / 2)
        z_up = scipy.stats.norm.ppf(1 - (1 - level) / 2)
        assert np.abs(np.abs(z_low) - np.abs(z_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE_notS + z_low * sd
        intervals_up = beta_MLE_notS + z_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        return coverage, intervals_up - intervals_low, intervals_low, intervals_up, nonzero, target

    # If no variable selected, no inference
    return None, None, None, None, None, None

def randomization_inference(X, Y, n, p, beta, groups, K=None,
                            weight_frac=1, level=0.9, solve_only=False):
    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    def estimate_K():
        loglike = rr.glm.poisson(X, counts=Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_full = restricted_estimator(loglike, np.array([True] * p))
        W_K = np.diag((Y - np.exp(X @ beta_full)) ** 2)
        return X.T @ W_K @ X

    def estimate_K_submodel():

        sigma_ = np.std(Y)
        # weights = dict([(i, 0.5) for i in np.unique(groups)])
        weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

        loglike = loglike = rr.glm.poisson(X, Y, quadratic=None)
        quad = rr.identity_quadratic(0,
                                     0,
                                     0,
                                     0)
        penalty = rr.group_lasso(groups,
                                 weights=weights,
                                 lagrange=1.)
        problem = rr.simple_problem(loglike, penalty)

        # if all groups are size 1, set up lasso penalty and run usual lasso solver... (see existing code)...
        solve_args = {'tol': 1.e-15, 'min_its': 100}
        observed_soln = problem.solve(quad, **solve_args)

        W_K_E = np.diag(np.exp(X @ observed_soln))
        return X.T @ W_K_E @ X

    def estimate_H():
        loglike = rr.glm.poisson(X, counts=Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_full = restricted_estimator(loglike, np.array([True] * p))
        W_H = np.diag(np.exp(X @ beta_full))
        return X.T @ W_H @ X

    if K is None:
        # print("(MLE) K estimated with full model")
        K = estimate_K()
        K_sub = estimate_K_submodel()
        H = estimate_H()
        print("K_sub norm", np.linalg.norm(K_sub, 'fro'))
        print("K norm", np.linalg.norm(K, 'fro'))
        print("K-K_sub norm", np.linalg.norm(K-K_sub, 'fro'))

    def estimate_hess():
        loglike = rr.glm.poisson(X, counts=Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_full = restricted_estimator(loglike, np.array([True] * p))
        W_H = np.diag(np.exp(X @ beta_full))
        return X.T @ W_H @ X
    hess = estimate_hess()

    sigma_ = np.std(Y)
    # weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    randomizer_scale = 1.
    conv = group_lasso_quasi.quasipoisson(X=X,
                                          counts=Y,
                                          groups=groups,
                                          weights=weights,
                                          useJacobian=True,
                                          ridge_term=0.,
                                          # perturb=np.zeros(p),
                                          # cov_rand=K_sub
                                          cov_rand=hess
                                          # randomizer_scale=randomizer_scale * sigma_
                                          )

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        Y_mean = np.exp(X.dot(beta))

        loglike = rr.glm.poisson(X, counts=Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    # Return the selected variables if we only want to solve the problem
    if solve_only:
        return None, None, solve_target_restricted(), nonzero, None, None

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)

        cov_score = conv.K

        """target_spec = full_targets_quasi(loglike=conv.loglike,
                                             solution=conv.observed_soln,
                                             cov_score=cov_score,
                                             dispersion=1)"""
        target_spec = selected_targets_quasi(loglike=conv.loglike,
                                             solution=conv.observed_soln,
                                             cov_score=cov_score,
                                             dispersion=1)
        # Trial with poisson targets
        """target_spec = selected_targets(loglike=conv.loglike,
                                             solution=conv.observed_soln,
                                             #cov_score=cov_score,
                                             dispersion=1)"""

        result = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        def true_K_H(beta_true):
            W_K = np.diag((Y - np.exp(X[:,nonzero] @ beta_true)) ** 2)
            W_H = np.diag(np.exp(X[:,nonzero] @ beta_true))

            return X[:,nonzero].T @ W_K @ X[:,nonzero], X[:,nonzero].T @ W_H @ X[:,nonzero]

        """KEE_true, HEE_true = true_K_H(beta_target)
        print("K, H, beta_MLE estimation checking for randomized MLE:")
        print("K_EE norm", np.linalg.norm(KEE_true, 'fro'))
        print("H_EE norm", np.linalg.norm(HEE_true, 'fro'))
        print("H_EE-KEE norm", np.linalg.norm(HEE_true-KEE_true, 'fro'))
        print("H_EE_inv -cov_target norm",
              np.linalg.norm(np.linalg.inv(HEE_true) - np.linalg.inv(HEE_true) @ KEE_true @ np.linalg.inv(HEE_true), 'fro'))
        K_EE_hat = cov_score[nonzero,nonzero]
        H_EE_hat = conv._hessian
        print("K_EE_hat norm", np.linalg.norm(K_EE_hat, 'fro'))
        print("H_EE_hat norm", np.linalg.norm(, 'fro'))"""

        K_hat_sub = conv.K
        K_hat_full = conv.K_full
        H_hat = conv.hessian

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, nonzero, \
               intervals[:, 0], intervals[:, 1], K_hat_sub, K_hat_full, H_hat

    return None, None, None, None, None, None, None, None, None

# Remain to be implemented
def randomization_inference_fast(X, Y, n, p, beta, groups, proportion=0.5, cov_rand=None,
                                 weight_frac=1, level=0.9, solve_only=False):
    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    if cov_rand is None:
        def estimate_hess():
            loglike = rr.glm.poisson(X, counts=Y)
            # For LASSO, this is the OLS solution on X_{E,U}
            beta_full = restricted_estimator(loglike, np.array([True] * p))
            W_H = np.diag(np.exp(X @ beta_full))
            return X.T @ W_H @ X * (1-proportion) / proportion

        hess = estimate_hess()
        cov_rand = hess

    sigma_ = np.std(Y)
    # weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = split_group_lasso_quasi.quasipoisson(X=X,
                                                counts=Y,
                                                groups=groups,
                                                weights=weights,
                                                useJacobian=True,
                                                proportion=proportion,
                                                cov_rand=cov_rand)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        Y_mean = np.exp(X.dot(beta))

        loglike = rr.glm.poisson(X, counts=Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    # Return the selected variables if we only want to solve the problem
    if solve_only:
        return None, None, solve_target_restricted(), nonzero, None, None

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)
        cov_score = conv._unscaled_cov_score

        target_spec = selected_targets_quasi(conv.loglike,
                                             conv.observed_soln,
                                             cov_score=cov_score,
                                             dispersion=1)

        result = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, nonzero, \
               intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None

# Remain to be implemented
def split_inference(X, Y, n, p, beta, groups, const,
                    weight_frac=1, proportion=0.5, level=0.9):
    ## selective inference with data carving

    sigma_ = np.std(Y)
    # weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = const(X=X,
                 counts=Y,
                 groups=groups,
                 weights=weights,
                 proportion=proportion,
                 useJacobian=True)

    signs, _ = conv.fit()
    nonzero = signs != 0

    # print("Carving selection", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():

        Y_mean = np.exp(X.dot(beta))

        loglike = rr.glm.poisson(X, counts=Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        conv.setup_inference(dispersion=1)
        cov_score = conv._unscaled_cov_score

        target_spec = selected_targets_quasi(conv.loglike,
                                             conv.observed_soln,
                                             cov_score=cov_score,
                                             dispersion=1)

        result = conv.inference(target_spec,
                                'selective_MLE',
                                level=level)
        estimate = result['MLE']
        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        K = ((1 - proportion) / proportion) * conv._unscaled_cov_score  # K matrix

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, conv._selection_idx, K, intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None, None, None


def data_splitting(X, Y, n, p, const, beta, nonzero=None, subset_select=None, groups=None,
                   weight_frac=1., proportion=0.67, level=0.9):
    if (nonzero is None) or (subset_select is None):
        # print("(Data Splitting) Selection done without carving")
        pi_s = proportion
        subset_select = np.zeros(n, np.bool)
        subset_select[:int(pi_s * n)] = True
        n1 = subset_select.sum()
        n2 = n - n1
        np.random.shuffle(subset_select)
        X_S = X[subset_select, :]
        Y_S = Y[subset_select]

        # Selection on the first subset of data
        p = X.shape[1]
        sigma_ = np.std(Y_S)
        # weights = dict([(i, 0.5) for i in np.unique(groups)])
        weights = dict([(i, (n1/n)*weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

        conv = const(X=X_S,
                     counts=Y_S,
                     groups=groups,
                     weights=weights,
                     useJacobian=True,
                     perturb=np.zeros(p),
                     ridge_term=0.)

        signs, _ = conv.fit()
        # print("signs",  signs)
        nonzero = signs != 0

    n1 = subset_select.sum()
    n2 = n - n1

    # print("Data splitting |E|:", nonzero.sum())

    if nonzero.sum() > 0:
        # Solving the inferential target
        def solve_target_restricted():
            Y_mean = np.exp(X.dot(beta))

            loglike = rr.glm.poisson(X, counts=Y_mean)
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # E: nonzero flag

        X_notS_E = X_notS[:, nonzero]

        loglike = rr.glm.poisson(X_notS, counts=Y_notS)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE_notS = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        W_H_notS = np.diag(np.exp(X_notS_E @ beta_MLE_notS))
        W_K_notS = np.diag((Y_notS - np.exp(X_notS_E @ beta_MLE_notS))** 2)

        H_EE = X_notS_E.T @ W_H_notS @ X_notS_E# * (n/n2)
        H_EE_inv = np.linalg.inv(H_EE)
        K_EE = X_notS_E.T @ W_K_notS @ X_notS_E# * (n/n2)
        cov = H_EE_inv @ K_EE @ H_EE_inv

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        z_low = scipy.stats.norm.ppf((1 - level) / 2)
        z_up = scipy.stats.norm.ppf(1 - (1 - level) / 2)
        assert np.abs(np.abs(z_low) - np.abs(z_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE_notS + z_low * sd
        intervals_up = beta_MLE_notS + z_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        return coverage, intervals_up - intervals_low, intervals_low, intervals_up, nonzero, target

    # If no variable selected, no inference
    return None, None, None, None, None, None

def test_comparison_quasipoisson_group_lasso(n=500,
                                             p=200,
                                             signal_fac=0.1,
                                             s=5,
                                             rho=0.3,
                                             randomizer_scale=1.,
                                             level=0.90,
                                             iter=50):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["beta size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []

    confint_df = pd.DataFrame()

    for signal_fac in [0.01, 0.03, 0.06, 0.1]:  # [0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):

            # np.random.seed(i)

            inst = quasi_poisson_group_instance
            const = group_lasso_quasi.quasipoisson
            const_split = split_group_lasso_quasi.quasipoisson

            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal, decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=4,
                                  nlevels=5,
                                  sdiscrete=2,  # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]

                n, p = X.shape

                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, K, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.67)
                    noselection = (coverage_s is None)

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups, K=K)

                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    # selection_idx_s; FIT GROUP lasso without randomization
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                        beta_target_naive= \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level, nonzero_true=(beta != 0))
                    noselection = (coverage_naive is None)

                if not noselection:
                    # F1 scores
                    F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_s)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

                    # Hessian MLE coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum()) * i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame([signal_str] * nonzero.sum()),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_MLE], axis=0)

                    # Carving coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["F1 score"].append(F1_s)
                    oper_char["method"].append('Carving')
                    df_s = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                      pd.DataFrame(beta_target_s),
                                      pd.DataFrame(conf_low_s),
                                      pd.DataFrame(conf_up_s),
                                      pd.DataFrame(beta[nonzero_s] != 0),
                                      pd.DataFrame([signal_str] * nonzero_s.sum()),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * F1_s),
                                      pd.DataFrame(["Carving"] * nonzero_s.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_s], axis=0)

                    # Data splitting coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                       pd.DataFrame(beta_target_s),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_s] != 0),
                                       pd.DataFrame([signal_str] * nonzero_s.sum()),
                                       pd.DataFrame(np.ones(nonzero_s.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_s.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    # Naive coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                          pd.DataFrame(beta_target_naive),
                                          pd.DataFrame(conf_low_naive),
                                          pd.DataFrame(conf_up_naive),
                                          pd.DataFrame(beta[nonzero_naive] != 0),
                                          pd.DataFrame([signal_str] * nonzero_naive.sum()),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                          pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                          ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/tests/quasipois_vary_signal.csv', index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['beta size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('selectinf/randomized/tests/quasipois_CI_vary_signal.csv', index=False)

    #sns.histplot(oper_char_df["beta size"])
    #plt.show()

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['beta size', 'method']).mean())

    # cov_plot = \
    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["beta size"],
                hue=oper_char_df["method"],
                showmeans=True,
                orient="v")
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["beta size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    #len_plot.set_ylim(5, 17)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["beta size"],
                          hue=oper_char_df["method"],
                          showmeans=True,
                          orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()

def test_comparison_quasipoisson_group_lasso_vary_s(n=1000,
                                                    p=100,
                                                    signal_fac=0.1,
                                                    s=5,
                                                    sigma=2,
                                                    rho=0.3,
                                                    randomizer_scale=1.,
                                                    full_dispersion=True,
                                                    level=0.90,
                                                    iter=50):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["sparsity size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    #oper_char["runtime"] = []


    confint_df = pd.DataFrame()

    for s in [5, 8, 10]:  # [0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):
            # np.random.seed(i)

            inst = quasi_poisson_group_instance
            inst_p = poisson_group_instance
            const = group_lasso_quasi.quasipoisson
            const_p = group_lasso.poisson
            const_split = split_group_lasso_quasi.quasipoisson

            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal, decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(25).repeat(4)

                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=10,
                                  nlevels=5,
                                  sdiscrete=s-3, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  phi=1.5,
                                  random_signs=True,
                                  center=False,
                                  scale=True)[:3]

                # print(X)

                """X, Y, beta = inst_p(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=0,
                                  nlevels=0,
                                  sdiscrete=0,  # s-3, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True,
                                  center=False,
                                  scale=True)[:3]"""

                n, p = X.shape

                noselection = False  # flag for a certain method having an empty selected set

                """
                if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, K, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.5)

                    noselection = (coverage_s is None)
                    if noselection:
                        print('No selection for carving')
                """
                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.67,
                                                     beta=beta, groups=groups, cov_rand=None)

                    noselection = (coverage is None)
                    print("MLE noselection", noselection)


                """# Poisson inference, to be deleted
                if not noselection:
                    # MLE inference (Poisson)
                    coverage_p, length_p, beta_target_p, nonzero_p, conf_low_p, conf_up_p = \
                        randomization_inference_poisson(X=X, Y=Y, n=n, p=p, #proportion=0.5,
                                                        beta=beta, groups=groups)
                    noselection = (coverage_p is None)"""

                """# Poisson data splitting, to be deleted
                if not noselection:
                    # data splitting poisson
                    coverage_dsp, lengths_dsp, conf_low_dsp, conf_up_dsp, nonzero_dsp, beta_target_dsp = \
                        data_splitting_poisson(X=X, Y=Y, n=n, p=p, const=const_p, groups=groups,
                                               beta=beta, proportion=0.67, level=0.9)
                    noselection = (coverage_dsp is None)"""

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, const=const, groups=groups, beta=beta,
                                       proportion=0.67, level=0.9)
                    noselection = (coverage_ds is None)
                    print("Data splitting noselection", noselection)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                        beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)
                    print("Naive noselection", noselection)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)
                    #F1_p = calculate_F1_score(beta, selection=nonzero_p)
                    #F1_dsp = calculate_F1_score(beta, selection=nonzero_dsp)

                    # MLE coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum()) * i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame(np.ones(nonzero.sum()) * s),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_MLE], axis=0)

                    """# Carving coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["F1 score"].append(F1_s)
                    oper_char["method"].append('Carving')
                    #oper_char["runtime"].append(0)
                    df_s = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                      pd.DataFrame(beta_target_s),
                                      pd.DataFrame(conf_low_s),
                                      pd.DataFrame(conf_up_s),
                                      pd.DataFrame(beta[nonzero_s] != 0),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * s),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * F1_s),
                                      pd.DataFrame(["Carving"] * nonzero_s.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_s], axis=0)"""

                    """# MLE (Poisson) coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_p))
                    oper_char["avg length"].append(np.mean(length_p))
                    oper_char["F1 score"].append(F1_p)
                    oper_char["method"].append('MLE (Poisson)')
                    df_p = pd.concat([pd.DataFrame(np.ones(nonzero_p.sum()) * i),
                                      pd.DataFrame(beta_target_p),
                                      pd.DataFrame(conf_low_p),
                                      pd.DataFrame(conf_up_p),
                                      pd.DataFrame(beta[nonzero_p] != 0),
                                      pd.DataFrame(np.ones(nonzero_p.sum()) * s),
                                      pd.DataFrame(np.ones(nonzero_p.sum()) * F1_p),
                                      pd.DataFrame(["MLE (Poisson)"] * nonzero_p.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_p], axis=0)"""

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_ds.sum()) * i),
                                       pd.DataFrame(beta_target_ds),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_ds] != 0),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * s),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_ds.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    """# Data splitting (poisson) coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_dsp))
                    oper_char["avg length"].append(np.mean(lengths_dsp))
                    oper_char["F1 score"].append(F1_dsp)
                    oper_char["method"].append('Data splitting (Poisson)')
                    df_dsp = pd.concat([pd.DataFrame(np.ones(nonzero_dsp.sum()) * i),
                                       pd.DataFrame(beta_target_dsp),
                                       pd.DataFrame(conf_low_dsp),
                                       pd.DataFrame(conf_up_dsp),
                                       pd.DataFrame(beta[nonzero_dsp] != 0),
                                       pd.DataFrame(np.ones(nonzero_dsp.sum()) * s),
                                       pd.DataFrame(np.ones(nonzero_dsp.sum()) * F1_dsp),
                                       pd.DataFrame(["Data splitting (Poisson)"] * nonzero_dsp.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)"""

                    # Naive coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                          pd.DataFrame(beta_target_naive),
                                          pd.DataFrame(conf_low_naive),
                                          pd.DataFrame(conf_up_naive),
                                          pd.DataFrame(beta[nonzero_naive] != 0),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * s),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                          pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                          ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/tests/quasipois_vary_sparsity.csv', index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('selectinf/randomized/tests/quasipois_CI_vary_sparsity.csv', index=False)

    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['sparsity size', 'method']).mean())

    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["sparsity size"],
                hue=oper_char_df["method"],
                orient="v")
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    len_plot.set_ylim(0, 8)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["sparsity size"],
                          hue=oper_char_df["method"],
                          showmeans=True,
                          orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()

def test_plotting_H_K(path='selectinf/randomized/tests/H_K_EE_df_vary_sparsity.csv'):
    H_K_df = pd.read_csv(path)

    H_K_plot = sns.boxplot(H_K_df,
                           showmeans=True,
                           orient="v")
    H_K_plot.set_xticklabels(H_K_plot.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()


def test_plotting(path='selectinf/randomized/tests/quasipois_vary_sparsity.csv'):
    oper_char_df = pd.read_csv(path)
    # sns.histplot(oper_char_df["sparsity size"])
    # plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['sparsity size', 'method']).mean())

    cov_plot = sns.boxplot(y=oper_char_df["coverage rate"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax1,
                           linewidth=1)
    cov_plot.set(title='Coverage')
    cov_plot.set_ylim(0.6, 1.05)
    # plt.tight_layout()
    cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
    # ax1.set_ylabel("")  # remove y label, but keep ticks

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax2,
                           linewidth=1)
    len_plot.set(title='Length')
    # len_plot.set_ylim(0, 100)
    # len_plot.set_ylim(3.5, 7.8)
    # plt.tight_layout()
    # ax2.set_ylabel("")  # remove y label, but keep ticks

    handles, labels = ax2.get_legend_handles_labels()
    # fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           palette="pastel",
                           orient="v", ax=ax3,
                           linewidth=1)
    F1_plot.set(title='F1 score')

    cov_plot.legend_.remove()
    len_plot.legend_.remove()
    F1_plot.legend_.remove()

    plt.show()

def test_plotting_separate(path='selectinf/randomized/tests/quasipois_vary_sparsity.csv'):
    oper_char_df = pd.read_csv(path)

    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    def plot_naive():
        naive_flag = oper_char_df["method"] == 'Naive'
        print(np.sum(naive_flag))

        print("Mean coverage rate/length:")
        print(oper_char_df.groupby(['sparsity size', 'method']).mean())

        cov_plot = sns.boxplot(y=oper_char_df.loc[naive_flag, "coverage rate"],
                               x=oper_char_df.loc[naive_flag, "beta size"],
                               # hue=oper_char_df["method"],
                               #palette="pastel",
                               color='lightcoral',
                               orient="v",
                               linewidth=1)
        cov_plot.set(title='Coverage of Naive Inference')
        cov_plot.set_ylim(0.5, 1.05)
        # plt.tight_layout()
        cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        plt.show()

    def plot_comparison():
        cov_plot = sns.boxplot(y=oper_char_df["coverage rate"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               palette="pastel",
                               orient="v",
                               linewidth=1)
        cov_plot.set(title='Coverage')
        cov_plot.set_ylim(0.5, 1.05)
        # plt.tight_layout()
        cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        cov_plot.legend(loc='lower center', ncol=3)
        plt.tight_layout()

        """
        for i in [2,5,8,11]:
            mybox = cov_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        """
        leg = cov_plot.get_legend()
        #leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    def plot_len_comparison():
        len_plot = sns.boxplot(y=oper_char_df["avg length"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               palette="pastel",
                               orient="v",
                               linewidth=1)
        len_plot.set(title='Length')
        # len_plot.set_ylim(0, 100)
        len_plot.legend(loc='lower center', ncol=3)
        len_plot.set_ylim(2, 12)
        plt.tight_layout()

        """
        for i in [2,5,8,11]:
            mybox = len_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        """
        leg = len_plot.get_legend()
        #leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    def plot_F1_comparison():
        F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               palette="pastel",
                               orient="v",
                               linewidth=1)
        F1_plot.set(title='F1 score')
        # len_plot.set_ylim(0, 100)
        F1_plot.legend(loc='lower center', ncol=3)
        F1_plot.set_ylim(0, 1)
        plt.tight_layout()

        """
        for i in [2,5,8,11]:
            mybox = len_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        """
        leg = F1_plot.get_legend()
        #leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    def plot_MLE_runtime():
        plt.figure(figsize=(8, 5))
        MLE_flag = oper_char_df["method"] == 'MLE'

        runtime_plot = sns.boxplot(y=oper_char_df.loc[MLE_flag, "runtime"],
                                   x=oper_char_df.loc[MLE_flag, "sparsity size"],
                                   # hue=oper_char_df["method"],
                                   # palette="pastel",
                                   #color='lightcoral',
                                   color='lightskyblue',
                                   orient="v",
                                   linewidth=1)
        runtime_plot.set(title='Runtime in Seconds for MLE')
        runtime_plot.set_ylim(0, 1.)
        # plt.tight_layout()
        #runtime_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
        plt.show()

    #plot_naive()
    plot_comparison()
    plot_len_comparison()
    plot_F1_comparison()
    #plot_MLE_runtime()