import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import time

import regreg.api as rr

from ..group_lasso_query import (group_lasso,
                                 split_group_lasso)

from ...base import selected_targets
from selectinf.randomized.tests.instance import gaussian_group_instance

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
                    n, weight_frac=1.5, level=0.9):
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

    conv = const(X=X,
                 Y=Y,
                 groups=groups,
                 weights=weights,
                 useJacobian=False,
                 perturb=np.zeros(p),
                 ridge_term=0.)

    signs, _ = conv.fit()
    nonzero = signs != 0

    print('Naive selection', conv._ordered_groups)

    if nonzero.sum() > 0:
        # E: nonzero flag
        X_E = X[:, nonzero]
        E_size = nonzero.sum()

        print("Naive |E|:", E_size)

        loglike = rr.glm.gaussian(X, Y)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE = restricted_estimator(loglike, nonzero)

        def solve_target_restricted():
            Y_mean = X.dot(beta)
            loglike = rr.glm.gaussian(X, Y_mean)
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        f_info = X_E.T @ X_E
        cov = np.linalg.inv(f_info) * dispersion

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n-E_size)
        qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n-E_size)
        assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE + qt_low * sd
        intervals_up = beta_MLE + qt_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        return coverage, intervals_up - intervals_low, nonzero, intervals_low, intervals_up, target

    return None, None, None, None, None, None

def posterior_inference(X, Y, n, p, beta, groups,
                        randomizer_scale=1.,
                        weight_frac=1.5, level=0.9):

    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    ##estimate noise level in data

    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = group_lasso.gaussian(X=X,
                                Y=Y,
                                groups=groups,
                                weights=weights,
                                useJacobian=True,
                                randomizer_scale=randomizer_scale * sigma_)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    def solve_target_restricted():
        Y_mean = X.dot(beta)
        loglike = rr.glm.gaussian(X, Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        print("MLE |E|:", nonzero.sum())

        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        result = conv.inference(target_spec,
                                method='posterior',
                                level=level)

        # pval = result['pvalue']
        intervals = np.asarray(result[['lower_credible',
                                       'upper_credible']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None
def randomization_inference(X, Y, n, p, beta, groups, hess=None,
                            randomizer_scale=1.,
                            weight_frac=1.5, level=0.9, solve_only = False):

    hess = X.T @ X
    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    ##estimate noise level in data

    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = group_lasso.gaussian(X=X,
                                Y=Y,
                                groups=groups,
                                weights=weights,
                                useJacobian=True,
                                randomizer_scale=hess)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    def solve_target_restricted():
        Y_mean = X.dot(beta)
        loglike = rr.glm.gaussian(X, Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    # Return the selected variables if we only want to solve the problem
    if solve_only:
        return None,None,solve_target_restricted(),nonzero,None,None

    if nonzero.sum() > 0:
        print("MLE |E|:", nonzero.sum())

        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        result = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None

def randomization_inference_fast(X, Y, n, p, beta, proportion,
                                 groups, hess=None, weight_frac=1.5, level=0.9):

    hess = X.T @ X * (1 - proportion) / proportion
    ## Use split group lasso to solve the hessian-randomized MLE problem efficiently
    ## Selection is consistent with the MLE method with the hessian randomization covariance
    ## but inference is carried out as if data carving were intended
    ## The two inference approaches are asymptotically the same

    ##estimate noise level in data
    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = split_group_lasso.gaussian(X=X,
                                      Y=Y,
                                      groups=groups,
                                      weights=weights,
                                      useJacobian=True,
                                      proportion=proportion,
                                      cov_rand=hess)

    signs, _ = conv.fit()
    nonzero = (signs != 0)

    # print("MLE selection:", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        Y_mean = X.dot(beta)
        loglike = rr.glm.gaussian(X, Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        print("MLE |E|:", nonzero.sum())
        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        result = conv.inference(target_spec,
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None

def split_inference(X, Y, n, p, beta, groups, const,
                    weight_frac=1.5, proportion=0.5, level=0.9):

    ## selective inference with data carving

    ##estimate noise level in data
    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = const(X=X,
                 Y=Y,
                 groups=groups,
                 weights=weights,
                 proportion=proportion, # proportion of data used for selection (n1/n)
                 useJacobian=True)

    signs, _ = conv.fit()
    nonzero = signs != 0

    # print("Carving selection", conv._ordered_groups)

    # Solving the inferential target
    def solve_target_restricted():
        Y_mean = X.dot(beta)
        loglike = rr.glm.gaussian(X, Y_mean)
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(loglike,
                                                 nonzero)
        return _beta_unpenalized

    if nonzero.sum() > 0:
        print("Carving |E|:", nonzero.sum())
        conv.setup_inference(dispersion=dispersion)

        target_spec = selected_targets(conv.loglike,
                                       conv.observed_soln,
                                       dispersion=dispersion)

        result = conv.inference(target_spec,
                                'selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        hess = ((1 - proportion) / proportion) * conv._unscaled_cov_score  # hessian

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, conv._selection_idx, hess, intervals[:, 0], intervals[:, 1]

    return None, None, None, None, None, None, None, None

def data_splitting(X, Y, n, p, beta, groups, weight_frac=1.5,
                   nonzero=None, subset_select=None,
                   proportion=0.5, level=0.9):
    if (nonzero is None) or (subset_select is None):
        # print("(Poisson Data Splitting) Selection done without carving")
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

        conv = group_lasso.gaussian(X=X_S,
                                    Y=Y_S,
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
            Y_mean = X.dot(beta)
            loglike = rr.glm.gaussian(X, Y_mean)
            # For LASSO, this is the OLS solution on X_{E,U}
            _beta_unpenalized = restricted_estimator(loglike,
                                                     nonzero)
            return _beta_unpenalized

        target = solve_target_restricted()

        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # E: nonzero flag

        X_notS_E = X_notS[:, nonzero]
        E_size = nonzero.sum()

        # Solve for the unpenalized MLE
        loglike = rr.glm.gaussian(X_notS, Y_notS)
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE_notS = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        dispersion_notS_E = np.linalg.norm(Y_notS - X_notS_E @ beta_MLE_notS) ** 2 / (n2 - E_size)
        f_info = X_notS_E.T @ X_notS_E
        cov = np.linalg.inv(f_info) * dispersion_notS_E

        # Standard errors
        sd = np.sqrt(np.diag(cov))

        # Normal quantiles
        qt_low = scipy.stats.t.ppf((1 - level) / 2, df=n2 - E_size)
        qt_up = scipy.stats.t.ppf(1 - (1 - level) / 2, df=n2 - E_size)
        assert np.abs(np.abs(qt_low) - np.abs(qt_up)) < 10e-6

        # Construct confidence intervals
        intervals_low = beta_MLE_notS + qt_low * sd
        intervals_up = beta_MLE_notS + qt_up * sd

        coverage = (target > intervals_low) * (target < intervals_up)

        return coverage, intervals_up - intervals_low, intervals_low, intervals_up, nonzero, target

    # If no variable selected, no inference
    return None, None, None, None, None, None

def test_comparison_group_lasso(n=500,
                                p=200,
                                signal_fac=0.1,
                                s=5,
                                rho=0.3,
                                randomizer_scale=1.,
                                level=0.90,
                                iter=10):
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
    #oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for signal_fac in [0.01, 0.03, 0.06, 0.1]: #[0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):

            #np.random.seed(i)

            inst  = gaussian_group_instance
            const = group_lasso.gaussian
            const_split = split_group_lasso.gaussian

            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal,decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=4,
                                  nlevels=5,
                                  sdiscrete=2, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]

                n, p = X.shape

                noselection = False    # flag for a certain method having an empty selected set

                if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, hessian, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.5)
                    noselection = (coverage_s is None)

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.5,
                                                     beta=beta, groups=groups, hess=hessian)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive,\
                        beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
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
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum())*i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame([signal_str] * nonzero.sum()),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)

                    confint_df = pd.concat([confint_df, df_MLE], axis=0)
                    #oper_char["runtime"].append(0)

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
    oper_char_df.to_csv('selectinf/randomized/tests/gaussian_vary_signal.csv', index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['beta size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('selectinf/randomized/tests/gaussian_CI_vary_signal.csv', index=False)

    #sns.histplot(oper_char_df["beta size"])
    #plt.show()

    print("Mean coverage rate/length:")
    print(oper_char_df.groupby(['beta size', 'method']).mean())

    #cov_plot = \
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
    len_plot.set_ylim(5,17)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                           x=oper_char_df["beta size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()


def test_comparison_gaussian_lasso_vary_s(n=500,
                                           p=200,
                                           signal_fac=0.1,
                                           s=5,
                                           sigma=2,
                                           rho=0.3,
                                           randomizer_scale=1.,
                                           full_dispersion=True,
                                           level=0.90,
                                           iter=10):
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
    # oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for s in [5,8,10]: #[0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):
            #np.random.seed(i)

            inst, const, const_split = gaussian_group_instance, group_lasso.gaussian, \
                                       split_group_lasso.gaussian
            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal, decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=20,
                                  nlevels=5,
                                  sdiscrete=s-3,#s-3, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]
                #print(X)

                n, p = X.shape

                noselection = False  # flag for a certain method having an empty selected set

                """if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, hessian, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.67)

                    noselection = (coverage_s is None)"""

                if not noselection:
                    # MLE inference
                    start = time.perf_counter()
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.67,
                                                     beta=beta, groups=groups)
                    end = time.perf_counter()
                    MLE_runtime = end - start
                    #print(MLE_runtime)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups,
                                       proportion=0.67, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                    beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # Posterior inference
                    coverage_pos, length_pos, beta_target_pos, nonzero_pos, conf_low_pos, conf_up_pos = \
                        posterior_inference(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups)
                    noselection = (coverage_pos is None)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)
                    F1_pos = calculate_F1_score(beta, selection=nonzero_pos)

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
                    

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    #oper_char["runtime"].append(0)
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

                    # Posterior coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_pos))
                    oper_char["avg length"].append(np.mean(length_pos))
                    oper_char["F1 score"].append(F1_pos)
                    oper_char["method"].append('Posterior')
                    df_pos = pd.concat([pd.DataFrame(np.ones(nonzero_pos.sum()) * i),
                                        pd.DataFrame(beta_target_pos),
                                        pd.DataFrame(conf_low_pos),
                                        pd.DataFrame(conf_up_pos),
                                        pd.DataFrame(beta[nonzero_pos] != 0),
                                        pd.DataFrame(np.ones(nonzero_pos.sum()) * s),
                                        pd.DataFrame(np.ones(nonzero_pos.sum()) * F1_pos),
                                        pd.DataFrame(["Posterior"] * nonzero_pos.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_pos], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/tests/gaussian_vary_sparsity.csv',index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('selectinf/randomized/tests/gaussian_CI_vary_sparsity.csv', index=False)

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
    len_plot.set_ylim(5,17)
    plt.show()

    F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                          x=oper_char_df["sparsity size"],
                          hue=oper_char_df["method"],
                          showmeans=True,
                          orient="v")
    F1_plot.set_ylim(0, 1)
    plt.show()

def test_plotting(path='selectinf/randomized/tests/gaussian_vary_sparsity.csv'):
    oper_char_df = pd.read_csv(path)
    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12,5))

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
    #plt.tight_layout()
    cov_plot.axhline(y=0.9, color='k', linestyle='--', linewidth=1)
    #ax1.set_ylabel("")  # remove y label, but keep ticks

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

def test_plotting_separate(path='selectinf/randomized/tests/gaussian_vary_sparsity.csv'):
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
        #len_plot.set_ylim(15, 25)
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
