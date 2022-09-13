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
from selectinf.randomized.tests.instance import gaussian_group_instance
from selectinf.tests.instance import logistic_instance
from ...base import restricted_estimator
import scipy.stats

def naive_inference(X, Y, beta, const, n, level=0.9):

    p = X.shape[1]
    sigma_ = np.std(Y)
    W = 1#np.sqrt(2 * np.log(p)) * sigma_

    conv = const(X, Y, W, perturb=np.zeros(p))

    signs = conv.fit()
    nonzero = signs != 0

    # Solving the inferential target
    def solve_target_restricted():
        def pi(x):
            return 1 / (1 + np.exp(-x))

        Y_mean = pi(X.dot(beta))

        loglike_Mean = rr.glm.logistic(X, successes=Y_mean, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike_Mean,
                                                 nonzero)
        return _beta_unpenalized

    target = solve_target_restricted()

    if nonzero.sum() > 0:
        # E: nonzero flag

        X_E = X[:, nonzero]

        def pi_hess(x):
            return np.exp(x) / (1 + np.exp(x)) ** 2

        loglike = rr.glm.logistic(X, successes=Y, trials=np.ones(n))
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(pi_hess(X_E @ beta_MLE))

        f_info = X_E.T @ W @ X_E
        cov = np.linalg.inv(f_info)

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

        return coverage, intervals_up - intervals_low

    return None, None

def randomization_inference(X, Y, n, p, beta, const,
                            randomizer_scale, level=0.9, solve_only = False):

    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    sigma_ = np.std(Y)
    W = 1#np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

    conv = const(X,
                 Y,
                 W,
                 randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()
    nonzero = signs != 0

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

    # Return the selected variables if we only want to solve the problem
    if solve_only:
        return None,None,solve_target_restricted(),nonzero

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

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, nonzero

    return None, None, None, None


def data_splitting(X, Y, n, p, beta, const,
                   randomizer_scale, proportion,
                   subset_select=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # sample sizes
    pi_s = proportion
    n1 = int(pi_s * n)
    n2 = n - n1

    if subset_select is None:
        subset_select = np.zeros(n, np.bool)
        subset_select[:int(pi_s * n)] = True
        np.random.shuffle(subset_select)

    X_S = X[subset_select, :]
    Y_S = Y[subset_select]

    _, _, target, nonzero = randomization_inference(X=X_S, Y=Y_S, n=n1, p=p,
                                                    beta=beta, const=const,
                                                    randomizer_scale=randomizer_scale,
                                                    solve_only=True)
    if nonzero is not None:
        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # Change target to the one corresponding to data splitting selection
        return naive_inference(X=X_notS, Y=Y_notS, target=target, E=nonzero, n=n2)

    # If no variable selected, no inference
    return None, None


def test_comparison_logistic_lasso(n=500,
                                   p=200,
                                   signal_fac=0.1,
                             s=10,
                             sigma=2,
                             rho=0.5,
                             randomizer_scale=1.,
                             full_dispersion=True,
                             level=0.90,
                             iter=200):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["beta size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []

    for signal_fac in [0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):

            #np.random.seed(i)

            inst, const = logistic_instance, lasso.logistic
            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal,decimals=2))

            while True:  # run until we get some selection
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  s=s,
                                  equicorrelated=True,
                                  rho=rho,
                                  random_signs=True,
                                  scale=True)[:3]

                idx = np.arange(p)
                sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))

                #print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

                n, p = X.shape

                coverage, length, beta_target, nonzero = \
                    randomization_inference(X=X, Y=Y, n=n, p=p,
                                            beta=beta, const=const,
                                            randomizer_scale=randomizer_scale)

                if coverage is not None:

                    # naive inference
                    coverage_naive, lengths_naive = \
                        naive_inference(X=X, Y=Y, beta=beta, const=const,
                                        n=n, level=level)

                    if coverage_naive is None:
                        break

                    # MLE coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["method"].append('MLE')

                    # Naive coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["method"].append('naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)

    sns.histplot(oper_char_df["beta size"])
    plt.show()

    #cov_plot = \
    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["beta size"],
                hue=oper_char_df["method"],
                orient="v")
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                            x=oper_char_df["beta size"],
                            hue=oper_char_df["method"],
                            orient="v")
    len_plot.set_ylim(5,15)
    plt.show()


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
    #   Write out cancelations with K


def test_comparison_logistic_lasso_vary_s(n=500,
                                           p=200,
                                           signal_fac=0.03,
                                           s=5,
                                           sigma=2,
                                           rho=0.5,
                                           randomizer_scale=1.,
                                           full_dispersion=True,
                                           level=0.90,
                                           iter=100):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["sparsity size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []

    for s in [5,10,15,20,25,30]:
        for i in range(iter):

            #np.random.seed(i)

            inst, const = logistic_instance, lasso.logistic
            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal,decimals=2))

            while True:  # run until we get some selection
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  s=s,
                                  equicorrelated=True,
                                  rho=rho,
                                  random_signs=True,
                                  scale=True)[:3]

                idx = np.arange(p)
                sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))

                # print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

                n, p = X.shape

                coverage, length, beta_target, nonzero = \
                    randomization_inference(X=X, Y=Y, n=n, p=p,
                                            beta=beta, const=const,
                                            randomizer_scale=randomizer_scale)

                if coverage is not None:

                    # naive inference
                    coverage_naive, lengths_naive = \
                        naive_inference(X=X, Y=Y, beta=beta, const=const,
                                        n=n, level=level)

                    if coverage_naive is None:
                        break

                    # MLE coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["method"].append('MLE')

                    # Naive coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["method"].append('naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)

    # cov_plot = \
    sns.boxplot(y=oper_char_df["coverage rate"],
                x=oper_char_df["sparsity size"],
                hue=oper_char_df["method"],
                orient="v")
    plt.show()

    # len_plot = \
    sns.boxplot(y=oper_char_df["avg length"],
                x=oper_char_df["sparsity size"],
                hue=oper_char_df["method"],
                orient="v")
    plt.show()
