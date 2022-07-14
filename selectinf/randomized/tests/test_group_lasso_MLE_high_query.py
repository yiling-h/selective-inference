import numpy as np
import nose.tools as nt

import regreg.api as rr


from ..group_lasso_query import group_lasso

from ...base import (full_targets,
                     selected_targets,
                     debiased_targets)
from selectinf.randomized.tests.instance import gaussian_group_instance
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance,
                               cox_instance)

def test_selected_targets(n=500,
                             p=200,
                             signal_fac=0.1,
                             sgroup=3,
                             groups=np.arange(50).repeat(4),
                             sigma=3.,
                             rho=0.3,
                             randomizer_scale=1,
                             weight_frac=1.2,
                             level=0.90):
    while True:

        inst = gaussian_group_instance
        signal = np.sqrt(signal_fac * 2 * np.log(p))

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

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv.inference(target_spec,
                                    method='selective_MLE')

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            print(pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals)
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

def test_selected_targets_lasso(n=2000,
                                p=200,
                                signal_fac=1.2,
                                s=5,
                                sigma=2,
                                rho=0.7,
                                randomizer_scale=1.,
                                full_dispersion=True):
    """
        Compare to R randomized lasso
        """

    inst, const = gaussian_instance, group_lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
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

        W = dict([(i, 0.8 * np.sqrt(2 * np.log(p)) * sigma_) for i in np.unique(np.arange(p))])

        conv = const(X,
                     Y,
                     weights=W,
                     groups=np.arange(p),
                     useJacobian=False,
                     randomizer_scale=randomizer_scale * sigma_)

        signs, _ = conv.fit()
        nonzero = (signs != 0)
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:

            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = np.linalg.norm(Y - X[:, nonzero].dot(np.linalg.pinv(X[:, nonzero]).dot(Y))) ** 2 / (
                            n - nonzero.sum())

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv.inference(target_spec,
                                    method='selective_MLE')

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            print(pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals)
            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals

### UNMODIFIED FROM 'test_selective_MLE_high.py'
def test_instance():
    n, p, s = 500, 100, 5
    X = np.random.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:s] = np.sqrt(2 * np.log(p) / n)
    Y = X.dot(beta) + np.random.standard_normal(n)

    scale_ = np.std(Y)
    # uses noise of variance n * scale_ / 4 by default
    L = lasso.gaussian(X, Y, 3 * scale_ * np.sqrt(2 * np.log(p) * np.sqrt(n)))
    signs = L.fit()
    E = (signs != 0)

    M = E.copy()
    M[-3:] = 1
    dispersion = np.linalg.norm(Y - X[:, M].dot(np.linalg.pinv(X[:, M]).dot(Y))) ** 2 / (n - M.sum())

    L.setup_inference(dispersion=dispersion)

    target_spec = selected_targets(L.loglike,
                                   L.observed_soln,
                                   features=M,
                                   dispersion=dispersion)

    print("check shapes", target_spec.observed_target.shape, E.sum())

    result = L.selective_MLE(target_spec)[0]

    intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

    beta_target = np.linalg.pinv(X[:, M]).dot(X.dot(beta))

    coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

    return coverage


### UNMODIFIED FROM 'test_selective_MLE_high.py'
def test_selected_targets_disperse(n=500,
                                   p=100,
                                   s=5,
                                   sigma=1.,
                                   rho=0.4,
                                   randomizer_scale=1,
                                   full_dispersion=True):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = 1.

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

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
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = np.linalg.norm(Y - X[:,nonzero].dot(np.linalg.pinv(X[:,nonzero]).dot(Y))) ** 2 / (n - nonzero.sum())

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv._selective_MLE(target_spec)[0]

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals
