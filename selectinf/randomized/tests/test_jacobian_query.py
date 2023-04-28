import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt

import regreg.api as rr

from ..lasso import (lasso,
                     split_lasso)
from ..group_lasso_query import (group_lasso,
                                 split_group_lasso)

from ...base import (full_targets,
                     selected_targets,
                     debiased_targets)
from selectinf.randomized.tests.instance import (gaussian_group_instance,
                                                 logistic_group_instance,
                                                 poisson_group_instance)
from selectinf.tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance,
                               cox_instance)
from ...base import restricted_estimator
from scipy.optimize import minimize

def test_selected_targets_lasso(n=500,
                              p=200,
                              signal_fac=1.2,
                              s=5,
                              sigma=2,
                              rho=0.7,
                              randomizer_scale=1.,
                              full_dispersion=True,
                              level=0.90,
                              iter=100):
    """
    Compare to R randomized lasso
    """

    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst, const = gaussian_instance, lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=True,
                              rho=rho,
                              sigma=sigma,
                              random_signs=True)[:3]

            idx = np.arange(p)
            sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
            print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

            n, p = X.shape

            sigma_ = np.std(Y)
            W = 0.8 * np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

            conv = const(X,
                         Y,
                         W,
                         ridge_term=0.,
                         randomizer_scale=randomizer_scale * sigma_)

            signs = conv.fit()
            nonzero = signs != 0
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                if full_dispersion:
                    dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
                else:
                    dispersion = np.linalg.norm(Y - X[:,nonzero].dot(np.linalg.pinv(X[:,nonzero]).dot(Y))) ** 2 / (n - nonzero.sum())

                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

                conv.setup_inference(dispersion=dispersion)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=dispersion)

                result = conv.inference(target_spec,
                                        'selective_MLE',
                                        level=level)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])

                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break   # Go to next iteration if we have some selection

def test_split_lasso(n=2000,
                     p=200,
                     signal_fac=1.2,
                     s=5,
                     sigma=3,
                     target='selected',
                     rho=0.4,
                     proportion=0.7,
                     orthogonal=False,
                     level=0.90,
                     iter=100):
    """
    Test data splitting lasso
    """

    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst, const = gaussian_instance, split_lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=True,
                              rho=rho,
                              sigma=sigma,
                              random_signs=True)[:3]

            n, p = X.shape

            sigma_ = np.std(Y)
            W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

            conv = const(X,
                         Y,
                         W,
                         proportion=proportion)

            signs = conv.fit()
            nonzero = signs != 0

            if nonzero.sum() > 0:

                true_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

                conv.setup_inference(dispersion=None)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln)


                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=level)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                coverage = (true_target > intervals[:, 0]) * (true_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])

                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break   # Go to next iteration if we have some selection

def test_selected_targets_group_lasso(n=500,
                                     p=200,
                                     signal_fac=1.2,
                                     sgroup=3,
                                     groups=np.arange(50).repeat(4),
                                     sigma=3.,
                                     rho=0.3,
                                     randomizer_scale=1,
                                     weight_frac=1.2,
                                     level=0.90,
                                     iter=100):
    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst = gaussian_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              equicorrelated=False,
                              rho=rho,
                              sigma=sigma,
                              random_signs=True)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)
            if n > p:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = sigma_ ** 2

            sigma_ = np.sqrt(dispersion)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
            conv = group_lasso.gaussian(X,
                                        Y,
                                        groups=groups,
                                        weights=weights,
                                        useJacobian=False,
                                        randomizer_scale=randomizer_scale * sigma_)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

                conv.setup_inference(dispersion=dispersion)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=dispersion)

                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=0.90)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])

                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break   # Go to next iteration if we have some selection

def test_selected_targets_posterior_group_lasso(n=500,
                                     p=200,
                                     signal_fac=1.2,
                                     sgroup=3,
                                     groups=np.arange(50).repeat(4),
                                     sigma=3.,
                                     rho=0.3,
                                     randomizer_scale=1,
                                     weight_frac=1.2,
                                     level=0.90,
                                     iter=1):
    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst = gaussian_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              ndiscrete=0,
                              nlevels=0,
                              sdiscrete=0,  # s-3, # How many discrete rvs are not null
                              equicorrelated=False,
                              rho=rho,
                              sigma=sigma,
                              random_signs=True,
                              center=False,
                              scale=True)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)
            if n > p:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = sigma_ ** 2

            sigma_ = np.sqrt(dispersion)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
            conv = group_lasso.gaussian(X,
                                        Y,
                                        groups=groups,
                                        weights=weights,
                                        useJacobian=True,
                                        randomizer_scale=randomizer_scale * sigma_)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

                conv.setup_inference(dispersion=dispersion)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=dispersion)

                result = conv.inference(target_spec,
                                        method='posterior',
                                        level=0.90)

                # pval = result['pvalue']
                intervals = np.asarray(result[['lower_credible', 'upper_credible']])

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])

                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break   # Go to next iteration if we have some selection

def test_selected_targets_split_group_lasso(n=500,
                                             p=200,
                                             signal_fac=1.2,
                                             sgroup=3,
                                             groups=np.arange(50).repeat(4),
                                             sigma=3.,
                                             rho=0.3,
                                             proportion=0.5,
                                             weight_frac=1.2,
                                             level=0.90,
                                             iter=100):
    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst = gaussian_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              equicorrelated=False,
                              rho=rho,
                              sigma=sigma,
                              random_signs=True)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)
            if n > p:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = sigma_ ** 2

            sigma_ = np.sqrt(dispersion)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
            conv = split_group_lasso.gaussian(X,
                                              Y,
                                              groups=groups,
                                              weights=weights,
                                              proportion=proportion,
                                              useJacobian=True)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

                conv.setup_inference(dispersion=dispersion)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=dispersion)

                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=0.90)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])

                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break  # Go to next iteration if we have some selection

def test_selected_targets_logistic_lasso(n=500,
                                          p=200,
                                          signal_fac=1.2,
                                          s=15,
                                          sigma=2,
                                          rho=0.7,
                                          randomizer_scale=1.,
                                          full_dispersion=True,
                                          level=0.90,
                                          iter=100):
    """
    Compare to R randomized lasso
    """

    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst, const = logistic_instance, lasso.logistic
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=True)[:3]

            idx = np.arange(p)
            sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
            print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

            n, p = X.shape

            sigma_ = np.std(Y)
            W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

            conv = const(X,
                         Y,
                         W,
                         randomizer_scale=randomizer_scale * sigma_)

            signs = conv.fit()
            nonzero = signs != 0
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:
                conv.setup_inference(dispersion=1)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=1)

                result = conv.inference(target_spec,
                                        'selective_MLE',
                                        level=0.9)
                estimate = result['MLE']
                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence',
                                               'upper_confidence']])

                # Solving the inferential target
                def solve_target_restricted():
                    def pi(x):
                        return 1 / (1 + np.exp(-x))

                    Y_mean = pi(X.dot(beta))

                    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero)
                    return _beta_unpenalized

                beta_target = solve_target_restricted()

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])
    
                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break   # Go to next iteration if we have some selection


    ## 1. Present plots on coverages + lengths, vary signal strength
    #   (or use SNR which takes into account sigma)
    #     Plot empirical confidence intervals from simulation
    ## 2. Add naive inference.
    ## 3. Slides:
    #   explain simulation
    #   explain variable selection (logistic lasso/group lasso)
    #   inference: try simplify the math into big pictures: MLE algorithm (1-2 slides)
    #   show plots: coverage + length
    #   try both lasso + (group lasso)
    ## 4. Data carving:
    #   How to calculate variance after Taylor?
    #   Write out cancelatiions with K

def test_selected_targets_split_logistic_lasso(n=500,
                                               p=200,
                                               signal_fac=1.2,
                                               s=15,
                                               sigma=2,
                                               rho=0.7,
                                               proportion=.7,
                                               full_dispersion=True,
                                               level=0.90,
                                               iter=100):
    """
    Compare to R randomized lasso
    """

    cover = []
    len_ = []

    for i in range(iter):

        np.random.seed(i)

        inst, const = logistic_instance, split_lasso.logistic
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get somee selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              s=s,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=True)[:3]

            idx = np.arange(p)
            sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
            print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

            n, p = X.shape

            sigma_ = np.std(Y)
            W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

            conv = const(X,
                         Y,
                         W,
                         proportion=proportion)

            signs = conv.fit()
            nonzero = signs != 0
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:
                conv.setup_inference(dispersion=1)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=1)

                result = conv.inference(target_spec,
                                        'selective_MLE',
                                        level=level)
                estimate = result['MLE']
                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence',
                                               'upper_confidence']])

                # Solving the inferential target
                def solve_target_restricted():
                    def pi(x):
                        return 1 / (1 + np.exp(-x))

                    Y_mean = pi(X.dot(beta))

                    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero)
                    return _beta_unpenalized

                beta_target = solve_target_restricted()

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                cover.extend(coverage)
                len_.extend(intervals[:, 1] - intervals[:, 0])

                print("Coverage so far ", np.mean(cover), )
                print("Lengths so far ", np.mean(len_))

                break  # Go to next iteration if we have some selection

def test_selected_targets_group_logistic_lasso(n=500,
                                               p=200,
                                               signal_fac=1,#1.2
                                               sgroup=5,
                                               groups=np.arange(50).repeat(4),
                                               rho=0.3,
                                               weight_frac=1.,
                                               randomizer_scale=1,
                                               level=0.90,
                                               iter=100):
    # Operating characteristics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []

    for i in range(iter):

        np.random.seed(i)

        inst = logistic_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get some selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              ndiscrete=0,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=True)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

            conv = group_lasso.logistic(X=X,
                                        successes=Y,
                                        trials=np.ones(n),
                                        groups=groups,
                                        weights=weights,
                                        useJacobian=True,
                                        ridge_term=0.,
                                        randomizer_scale=randomizer_scale * sigma_)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                # Solving the inferential target
                def solve_target_restricted():
                    def pi(x):
                        return 1 / (1 + np.exp(-x))

                    Y_mean = pi(X.dot(beta))

                    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero)
                    return _beta_unpenalized

                conv.setup_inference(dispersion=1)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=1)

                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=level)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                beta_target = solve_target_restricted()

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                # MLE coverage
                oper_char["coverage rate"].append(np.mean(coverage))
                oper_char["avg length"].append(np.mean(intervals[:, 1] - intervals[:, 0]))

                print("Coverage so far ", np.mean(oper_char["coverage rate"]))
                print("Lengths so far ", np.mean(oper_char["avg length"]))
                #print(np.round(intervals[:, 0],1))
                #print(np.round(intervals[:, 1], 1))
                #print(np.round(beta_target, 1))

                break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)

    # cov_plot = \
    sns.boxplot(y=oper_char_df["coverage rate"],
                meanline=True,
                orient="v")
    plt.show()

def test_selected_targets_group_logistic_lasso_hessian(n=500,
                                                       p=200,
                                                       signal_fac=1.2,
                                                       sgroup=3,
                                                       groups=np.arange(50).repeat(4),
                                                       rho=0.3,
                                                       weight_frac=1.,
                                                       randomizer_scale=1,
                                                       level=0.90,
                                                       iter=100):
    # Operating characteristics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    # Operating characteristics
    oper_char_split = {}
    oper_char_split["coverage rate"] = []
    oper_char_split["avg length"] = []

    for i in range(iter):

        #np.random.seed(i)

        inst = logistic_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get some selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=True)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])


            # Hessian calculations
            proportion = 0.5
            conv_split = split_group_lasso.logistic(X,
                                                    successes=Y,
                                                    groups=groups,
                                                    weights=weights,
                                                    proportion=proportion,
                                                    useJacobian=True)

            signs_split, _ = conv_split.fit()
            nonzero_split = (signs_split != 0)
            hess = conv_split._unscaled_cov_score  # hessian

            conv = group_lasso.logistic(X=X,
                                        successes=Y,
                                        trials=np.ones(n),
                                        groups=groups,
                                        weights=weights,
                                        useJacobian=True,
                                        ridge_term=0.,
                                        cov_rand=hess)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0 and nonzero_split.sum() > 0:

                # Solving the inferential target
                def solve_target_restricted():
                    def pi(x):
                        return 1 / (1 + np.exp(-x))

                    Y_mean = pi(X.dot(beta))

                    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero)
                    return _beta_unpenalized

                # Solving the inferential target
                def solve_target_restricted_split():
                    def pi(x):
                        return 1 / (1 + np.exp(-x))

                    Y_mean = pi(X.dot(beta))

                    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero_split)
                    return _beta_unpenalized

                # non--split inference with hessian randomization
                conv.setup_inference(dispersion=1)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=1)

                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=level)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                beta_target = solve_target_restricted()

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                # MLE coverage
                oper_char["coverage rate"].append(np.mean(coverage))
                oper_char["avg length"].append(np.mean(intervals[:, 1] - intervals[:, 0]))

                # split inference with hessian randomization
                conv_split.setup_inference(dispersion=1)

                target_spec_split = selected_targets(conv_split.loglike,
                                                     conv_split.observed_soln,
                                                     dispersion=1)

                result_split = conv_split.inference(target_spec_split,
                                                    method='selective_MLE',
                                                    level=level)

                pval_split = result_split['pvalue']
                intervals_split = np.asarray(result_split[['lower_confidence', 'upper_confidence']])

                beta_target_split = solve_target_restricted_split()

                coverage_split = (beta_target_split > intervals_split[:, 0]) * \
                                 (beta_target_split < intervals_split[:, 1])

                # MLE coverage
                oper_char_split["coverage rate"].append(np.mean(coverage_split))
                oper_char_split["avg length"].append(np.mean(intervals_split[:, 1] - intervals_split[:, 0]))

                print("Coverage (nonsplit) so far ", np.mean(oper_char["coverage rate"]))
                print("Lengths (nonsplit) so far ", np.mean(oper_char["avg length"]))

                print("Coverage (split) so far ", np.mean(oper_char_split["coverage rate"]))
                print("Lengths (split) so far ", np.mean(oper_char_split["avg length"]))

                break  # Go to next iteration if we have some selection
    """
    oper_char_df = pd.DataFrame.from_dict(oper_char)

    # cov_plot = \
    sns.boxplot(y=oper_char_df["coverage rate"],
                meanline=True,
                orient="v")
    plt.show()
    """

def test_selected_targets_split_group_logistic_lasso(n=500,
                                                     p=200,
                                                     signal_fac=1.2,
                                                     sgroup=3,
                                                     groups=np.arange(50).repeat(4),
                                                     rho=0.3,
                                                     proportion=0.5,
                                                     weight_frac=1.,
                                                     randomizer_scale=1,
                                                     level=0.90,
                                                     iter=100):
    # Operating characteristics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []

    for i in range(iter):

        #np.random.seed(i)

        inst = logistic_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get some selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=True)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

            conv = split_group_lasso.logistic(X,
                                              successes=Y,
                                              groups=groups,
                                              weights=weights,
                                              proportion=proportion,
                                              useJacobian=True)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                # Solving the inferential target
                def solve_target_restricted():
                    def pi(x):
                        return 1 / (1 + np.exp(-x))

                    Y_mean = pi(X.dot(beta))

                    loglike = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero)
                    return _beta_unpenalized

                conv.setup_inference(dispersion=1)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=1)

                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=level)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                beta_target = solve_target_restricted()

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                # MLE coverage
                oper_char["coverage rate"].append(np.mean(coverage))
                oper_char["avg length"].append(np.mean(intervals[:, 1] - intervals[:, 0]))

                print("Coverage so far ", np.mean(oper_char["coverage rate"]))
                print("Lengths so far ", np.mean(oper_char["avg length"]))

                break  # Go to next iteration if we have some selection

def test_selected_targets_group_poisson_lasso(n=500,
                                              p=200,
                                              signal_fac=0.5,  # 1.2
                                              sgroup=5,
                                              groups=np.arange(50).repeat(4),
                                              rho=0.3,
                                              weight_frac=1.,
                                              randomizer_scale=1,
                                              level=0.90,
                                              iter=100):
    # Operating characteristics
    oper_char = {}
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []

    for i in range(iter):

        np.random.seed(i)

        inst = poisson_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

        while True:  # run until we get some selection
            X, Y, beta = inst(n=n,
                              p=p,
                              signal=signal,
                              sgroup=sgroup,
                              groups=groups,
                              ndiscrete=0,
                              sdiscrete=0,
                              equicorrelated=True,
                              rho=rho,
                              random_signs=False)[:3]

            n, p = X.shape

            ##estimate noise level in data

            sigma_ = np.std(Y)

            ##solve group LASSO with group penalty weights = weights

            weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

            conv = group_lasso.poisson(X=X,
                                       counts=Y,
                                       groups=groups,
                                       weights=weights,
                                       useJacobian=True,
                                       ridge_term=0.,
                                       randomizer_scale=randomizer_scale * sigma_)

            signs, _ = conv.fit()
            nonzero = (signs != 0)
            print("dimensions", n, p, nonzero.sum())

            if nonzero.sum() > 0:

                # Solving the inferential target
                def solve_target_restricted():

                    Y_mean = np.exp(X.dot(beta))

                    loglike = rr.glm.poisson(X, counts=Y_mean)
                    # For LASSO, this is the OLS solution on X_{E,U}
                    _beta_unpenalized = restricted_estimator(loglike,
                                                             nonzero)
                    return _beta_unpenalized

                conv.setup_inference(dispersion=1)

                target_spec = selected_targets(conv.loglike,
                                               conv.observed_soln,
                                               dispersion=1)

                result = conv.inference(target_spec,
                                        method='selective_MLE',
                                        level=level)

                pval = result['pvalue']
                intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

                beta_target = solve_target_restricted()

                coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

                # MLE coverage
                oper_char["coverage rate"].append(np.mean(coverage))
                oper_char["avg length"].append(np.mean(intervals[:, 1] - intervals[:, 0]))

                print("Coverage so far ", np.mean(oper_char["coverage rate"]))
                print("Lengths so far ", np.mean(oper_char["avg length"]))

                # Tabulate results
                d = {'target': beta_target,
                     'Selective MLE': result['MLE'],
                     'L_Poisson': intervals[:, 0], 'U_Poisson': intervals[:, 1],
                     'Coverage_P': coverage}
                df = pd.DataFrame(data=d)
                print(df)
                #print(np.round(intervals[:, 0],1))
                #print(np.round(intervals[:, 1], 1))
                #print(np.round(beta_target, 1))

                break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)

    # cov_plot = \
    # sns.boxplot(y=oper_char_df["coverage rate"],
    #             meanline=True,
    #             orient="v")
    # plt.show()