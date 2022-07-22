import numpy as np
import pandas as pd
import nose.tools as nt

import regreg.api as rr

from ..lasso import (lasso,
                     split_lasso)
from ..group_lasso_query import (group_lasso,
                                 split_group_lasso)

from ...base import (full_targets,
                     selected_targets,
                     debiased_targets)
from selectinf.randomized.tests.instance import gaussian_group_instance
from ...tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance,
                               cox_instance)

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

    op_char = np.zeros(shape=(iter, 3))

    for i in range(iter):

        inst, const = gaussian_instance, lasso.gaussian
        signal = np.sqrt(signal_fac * 2 * np.log(p))

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

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv.inference(target_spec,
                                    'selective_MLE',
                                    level=level)

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            op_char_i = np.array([nonzero.sum(),
                                  sum(coverage) / len(coverage),
                                  np.mean(intervals[:, 1] - intervals[:, 0])])
            op_char[i, :] = op_char_i

            #print(pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals)
            #return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals
    print(np.mean(op_char, axis=0))

def test_split_lasso(n=2000,
                     p=200,
                     signal_fac=1.2,
                     s=5,
                     sigma=3,
                     target='selected',
                     rho=0.4,
                     proportion=0.5,
                     orthogonal=False,
                     level=0.90,
                     iter=100):
    """
    Test data splitting lasso
    """

    op_char = np.zeros(shape=(iter, 3))

    for i in range(iter):

        inst, const = gaussian_instance, split_lasso.gaussian
        signal = np.sqrt(signal_fac * np.log(p))

        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        if orthogonal:
            X = np.linalg.svd(X, full_matrices=False)[0] * np.sqrt(n)
            Y = X.dot(beta) + np.random.standard_normal(n) * sigma

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(np.log(p)) * sigma_
        W[0] = 0

        conv = const(X,
                     Y,
                     W,
                     proportion)

        signs = conv.fit()
        nonzero = signs != 0

        if nonzero.sum() > 0:

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=sigma ** 2)

            conv.setup_inference(dispersion=sigma ** 2)

            result = conv.inference(target_spec,
                                    method='selective_MLE',
                                    level=level)

            true_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            coverage = (true_target > intervals[:, 0]) * (true_target < intervals[:, 1])

            op_char_i = np.array([nonzero.sum(),
                                  sum(coverage) / len(coverage),
                                  np.mean(intervals[:, 1] - intervals[:, 0])])
            op_char[i, :] = op_char_i

            #print(pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals)
            #return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals
    print(np.mean(op_char, axis=0))

def test_selected_targets_group_lasso(n=500,
                                     p=200,
                                     signal_fac=0.1,
                                     sgroup=3,
                                     groups=np.arange(50).repeat(4),
                                     sigma=3.,
                                     rho=0.3,
                                     randomizer_scale=1,
                                     weight_frac=1.2,
                                     level=0.90,
                                     iter=100):

    op_char = np.zeros(shape=(iter,3))

    for i in range(iter):

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
                                    method='selective_MLE',
                                    level=0.90)

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            op_char_i = np.array([nonzero.sum(),
                                  sum(coverage)/len(coverage),
                                  np.mean(intervals[:, 1] - intervals[:, 0])])
            op_char[i,:] = op_char_i
            #print(pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, sum(coverage)/len(coverage), intervals)
            #return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals
    print(np.mean(op_char, axis=0))


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

    op_char = np.zeros(shape=(iter,3))

    for i in range(iter):

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

            conv.setup_inference(dispersion=dispersion)

            target_spec = selected_targets(conv.loglike,
                                           conv.observed_soln,
                                           dispersion=dispersion)

            result = conv.inference(target_spec,
                                    method='selective_MLE',
                                    level=0.90)

            pval = result['pvalue']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            op_char_i = np.array([nonzero.sum(),
                                  sum(coverage)/len(coverage),
                                  np.mean(intervals[:, 1] - intervals[:, 0])])
            op_char[i,:] = op_char_i
            #print(pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, sum(coverage)/len(coverage), intervals)
            #return pval[beta[nonzero] == 0], pval[beta[nonzero] != 0], coverage, intervals
    print(np.mean(op_char, axis=0))
