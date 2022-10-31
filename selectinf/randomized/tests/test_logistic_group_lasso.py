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
                                                 logistic_group_instance)

from ...base import restricted_estimator
import scipy.stats

def naive_inference(X, Y, groups, beta, const,
                    n, weight_frac=1, level=0.9):

    p = X.shape[1]
    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = const(X=X,
                 successes=Y,
                 trials=np.ones(n),
                 groups=groups,
                 weights=weights,
                 useJacobian=True,
                 perturb=np.zeros(p),
                 ridge_term=0.)

    signs, _ = conv.fit()
    nonzero = signs != 0

    print('Naive selection', conv._ordered_groups)

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
                            groups, hess=None, randomizer_scale=None,
                            weight_frac=1, level=0.9, solve_only = False):

    ## solve_only: bool variable indicating whether
    ##              1) we only need the solver's output
    ##              or
    ##              2) we also want inferential results

    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

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

    print("MLE selection:", conv._ordered_groups)

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
                                method='selective_MLE',
                                level=level)

        pval = result['pvalue']
        intervals = np.asarray(result[['lower_confidence',
                                       'upper_confidence']])

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, nonzero

    return None, None, None, None

def split_inference(X, Y, n, p, beta, groups, const,
                    weight_frac=1, proportion=0.5, level=0.9):

    ## selective inference with data carving

    sigma_ = np.std(Y)
    #weights = dict([(i, 0.5) for i in np.unique(groups)])
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])

    conv = const(X=X,
                 successes=Y,
                 groups=groups,
                 weights=weights,
                 proportion=proportion,
                 useJacobian=True)

    signs, _ = conv.fit()
    nonzero = signs != 0

    print("Carving selection", conv._ordered_groups)

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

        beta_target = solve_target_restricted()

        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

        hess = conv._unscaled_cov_score  # hessian

        return coverage, (intervals[:, 1] - intervals[:, 0]), beta_target, \
               nonzero, conv._selection_idx, hess

    return None, None, None, None, None, None

def data_splitting(X, Y, n, p, beta, nonzero,
                   subset_select=None, level=0.9):
    n1 = subset_select.sum()
    n2 = n - n1

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

        target = solve_target_restricted()

        X_notS = X[~subset_select, :]
        Y_notS = Y[~subset_select]

        # E: nonzero flag

        X_notS_E = X_notS[:, nonzero]

        # Solve for the unpenalized MLE
        def pi_hess(x):
            return np.exp(x) / (1 + np.exp(x)) ** 2

        loglike = rr.glm.logistic(X_notS, successes=Y_notS, trials=np.ones(n2))
        # For LASSO, this is the OLS solution on X_{E,U}
        beta_MLE_notS = restricted_estimator(loglike, nonzero)

        # Calculation the asymptotic covariance of the MLE
        W = np.diag(pi_hess(X_notS_E @ beta_MLE_notS))

        f_info = X_notS_E.T @ W @ X_notS_E
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

        return coverage, intervals_up - intervals_low

    # If no variable selected, no inference
    return None, None

def test_comparison_logistic_group_lasso(n=500,
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

    for signal_fac in [0.01, 0.03, 0.06, 0.1]: #[0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):

            np.random.seed(i)

            inst  = logistic_group_instance
            const = group_lasso.logistic
            const_split = split_group_lasso.logistic

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
                    coverage_s, length_s, beta_target_s, nonzero_s, selection_idx_s, hessian = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.5)

                    noselection = (coverage_s is None)

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero = \
                        randomization_inference(X=X, Y=Y, n=n, p=p,
                                                beta=beta, const=const,
                                                groups=groups, hess=hessian)

                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)

                if not noselection:

                    # Hessian MLE coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["method"].append('MLE')

                    """
                    # Carving coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["method"].append('Carving')
                    """

                    # Data splitting coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["method"].append('Data splitting')

                    # Naive coverage
                    oper_char["beta size"].append(signal_str)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["method"].append('Naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/tests/oper_char_vary_signal.csv', index=False)

    sns.histplot(oper_char_df["beta size"])
    plt.show()

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


def test_comparison_logistic_lasso_vary_s(n=500,
                                           p=200,
                                           signal_fac=0.1,
                                           s=5,
                                           sigma=2,
                                           rho=0.5,
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

    for s in [5,8,10,15]: #[0.01, 0.03, 0.06, 0.1]:
        for i in range(iter):
            # np.random.seed(i)

            inst, const, const_split = logistic_group_instance, group_lasso.logistic, \
                                       split_group_lasso.logistic
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

                if not noselection:
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, selection_idx_s, hessian = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.5)

                    noselection = (coverage_s is None)
                    if noselection:
                        print('No selection for carving')

                if not noselection:
                    # MLE inference
                    coverage, length, beta_target, nonzero = \
                        randomization_inference(X=X, Y=Y, n=n, p=p,
                                                beta=beta, const=const,
                                                groups=groups, hess=hessian)

                    noselection = (coverage is None)
                    if noselection:
                        print('No selection for MLE')

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
                    noselection = (coverage_ds is None)
                    if noselection:
                        print('No selection for data splitting')

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)
                    if noselection:
                        print('No selection for naive')

                if not noselection:

                    # MLE coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["method"].append('MLE')

                    """
                    # Carving coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["method"].append('Carving')
                    """

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["method"].append('Data splitting')

                    # Naive coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["method"].append('Naive')

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('selectinf/randomized/tests/oper_char_vary_s.csv',index=False)

    sns.histplot(oper_char_df["sparsity size"])
    plt.show()

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

def test_plotting(path='selectinf/randomized/tests/oper_char_vary_s.csv'):
    oper_char_df = pd.read_csv(path)
    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

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
    #len_plot.set_ylim(0, 100)
    len_plot.set_ylim(7, 17)
    #plt.tight_layout()
    #ax2.set_ylabel("")  # remove y label, but keep ticks

    handles, labels = ax2.get_legend_handles_labels()
    #fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.2)
    fig.subplots_adjust(bottom=0.2)
    fig.legend(handles, labels, loc='lower center', ncol=4)

    cov_plot.legend_.remove()
    len_plot.legend_.remove()

    plt.show()

def test_plotting_separate(path='selectinf/randomized/tests/oper_char_vary_s.csv'):
    oper_char_df = pd.read_csv(path)
    naive_flag = oper_char_df["method"] == "Na$\ddot{i}$ve"
    oper_char_df.loc[naive_flag, "method"] = 'Naive'

    #sns.histplot(oper_char_df["sparsity size"])
    #plt.show()

    def plot_naive():
        naive_flag = oper_char_df["method"] == 'Naive'

        print("Mean coverage rate/length:")
        print(oper_char_df.groupby(['sparsity size', 'method']).mean())

        cov_plot = sns.boxplot(y=oper_char_df.loc[naive_flag, "coverage rate"],
                               x=oper_char_df.loc[naive_flag, "sparsity size"],
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

        for i in [2,5,8,11]:
            mybox = cov_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        leg = cov_plot.get_legend()
        leg.legendHandles[2].set_color('lightcoral')
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
        len_plot.set_ylim(5, 18)
        plt.tight_layout()

        for i in [2,5,8,11]:
            mybox = len_plot.artists[i]
            mybox.set_facecolor('lightcoral')
        leg = len_plot.get_legend()
        leg.legendHandles[2].set_color('lightcoral')
        plt.show()

    plot_naive()
    plot_comparison()
    plot_len_comparison()
