import numpy as np
from scipy.stats import norm as ndist

from ..multitask_lasso import multi_task_lasso
from ...tests.instance import gaussian_multitask_instance

def test_multitask_lasso_hetero(ntask=2,
                                nsamples=500 * np.ones(2),
                                p=100,
                                global_sparsity=.8,
                                task_sparsity=.3,
                                sigma=1. * np.ones(2),
                                signal_fac=0.5,
                                rhos=0. * np.ones(2),
                                weight=2.,
                                randomizer_variation = 1.):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:

        response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                           nsamples,
                                                                                           p,
                                                                                           global_sparsity,
                                                                                           task_sparsity,
                                                                                           sigma,
                                                                                           signal,
                                                                                           rhos,
                                                                                           random_signs=True,
                                                                                           equicorrelated=False)[:4]

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma
        randomizer_scales = 0.71 * np.array([sigmas_[i] for i in range(ntask)])

        _initial_omega = np.array([randomizer_scales[i] * _gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T

        multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                response_vars,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=randomizer_scales,
                                                perturbations=None)

        active_signs = multi_lasso.fit(perturbations=_initial_omega)

        if (active_signs != 0).sum() > 0:

            dispersions = sigma ** 2

            estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)

            beta_target = []

            for i in range(ntask):
                X, y = multi_lasso.loglikes[i].data
                beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i])))

            beta_target = np.asarray(beta_target)
            pivot_ = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))
            pivot = 2 * np.minimum(pivot_, 1. - pivot_)

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            return coverage, intervals[:, 1] - intervals[:, 0], pivot


def test_multitask_lasso_naive_hetero(ntask=2,
                                      nsamples=500 * np.ones(2),
                                      p=100,
                                      global_sparsity=.8,
                                      task_sparsity=.3,
                                      sigma=1. * np.ones(2),
                                      signal_fac=0.5,
                                      rhos=0. * np.ones(2),
                                      weight=2.):

    nsamples = nsamples.astype(int)
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        response_vars, predictor_vars, beta = gaussian_multitask_instance(ntask,
                                                                          nsamples,
                                                                          p,
                                                                          global_sparsity,
                                                                          task_sparsity,
                                                                          sigma,
                                                                          signal,
                                                                          rhos,
                                                                          random_signs=True,
                                                                          equicorrelated=False)[:3]

        feature_weight = weight * np.ones(p)

        sigmas_ = sigma

        perturbations = np.zeros((p, ntask))

        multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                                response_vars,
                                                feature_weight,
                                                ridge_term=None,
                                                randomizer_scales=1. * sigmas_,
                                                perturbations=perturbations)
        active_signs = multi_lasso.fit()

        dispersions = sigma ** 2

        coverage = []
        pivot = []

        if (active_signs != 0).sum() > 0:

            for i in range(ntask):

                X, y = multi_lasso.loglikes[i].data
                beta_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i]))
                Qfeat = np.linalg.inv(X[:, (active_signs[:, i] != 0)].T.dot(X[:, (active_signs[:, i] != 0)]))
                observed_target = np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(y)
                cov_target = Qfeat * dispersions[i]

                alpha = 1. - 0.90
                quantile = ndist.ppf(1 - alpha / 2.)

                intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                       observed_target + quantile * np.sqrt(np.diag(cov_target))]).T

                coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))

                pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
                pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], np.asarray(pivot)


import matplotlib as mpl
mpl.use('tkagg')
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

def test_coverage(nsim=100, coverage=False):

    if coverage == False:

        pivots_sel = []
        pivots_naive = []

        for n in range(nsim):

            ntask = 4

            pivot_sel = test_multitask_lasso_hetero(ntask=ntask,
                                                    nsamples=1000 * np.ones(ntask),
                                                    p=100,
                                                    global_sparsity=0.95,
                                                    task_sparsity=0.50,
                                                    sigma=1. * np.ones(ntask),
                                                    signal_fac=np.array([0.5, 1.]),
                                                    rhos=0.70 * np.ones(ntask),
                                                    weight=2.)[2]

            pivot_naive = test_multitask_lasso_naive_hetero(ntask=ntask,
                                                            nsamples=1000 * np.ones(ntask),
                                                            p=100,
                                                            global_sparsity=0.95,
                                                            task_sparsity=0.50,
                                                            sigma=1. * np.ones(ntask),
                                                            signal_fac=np.array([0.5, 1.]),
                                                            rhos=0.70 * np.ones(ntask),
                                                            weight=2.)[2]

            pivots_sel.extend(pivot_sel)
            pivots_naive.extend(pivot_naive)

            print("iteration completed ", n)

        plt.clf()

        ecdf_MLE = ECDF(np.asarray(pivots_sel))
        ecdf_naive = ECDF(np.asarray(pivots_naive))

        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
        plt.plot(grid, ecdf_naive(grid), c='red', marker='^')
        plt.plot(grid, grid, 'k--')
        plt.show()

    if coverage == True:

        covs_sel = []
        lens_sel = []
        pivots_sel = []

        covs_naive = []
        lens_naive = []
        pivots_naive = []

        for n in range(nsim):
            ntask = 4

            cov_sel, len_sel, pivot_sel = test_multitask_lasso_hetero(ntask=ntask,
                                                                      nsamples=1000 * np.ones(ntask),
                                                                      p=100,
                                                                      global_sparsity=0.95,
                                                                      task_sparsity=0.50,
                                                                      sigma=1. * np.ones(ntask),
                                                                      signal_fac=np.array([0.5, 1.]),
                                                                      rhos=0.70 * np.ones(ntask),
                                                                      weight=2.)

            cov_naive, len_naive, pivot_naive = test_multitask_lasso_naive_hetero(ntask=ntask,
                                                                                  nsamples=1000 * np.ones(ntask),
                                                                                  p=100,
                                                                                  global_sparsity=0.95,
                                                                                  task_sparsity=0.50,
                                                                                  sigma=1. * np.ones(ntask),
                                                                                  signal_fac=np.array([0.5, 1.]),
                                                                                  rhos=0.70 * np.ones(ntask),
                                                                                  weight=2.)[2]

            covs_sel.extend(cov_sel)
            lens_sel.extend(len_sel)
            pivots_sel.extend(pivot_sel)

            covs_naive.extend(cov_naive)
            lens_naive.extend(len_naive)
            pivots_naive.extend(pivot_naive)

            print("iteration completed ", n)
            print("coverage so far ", np.mean(np.asarray(covs_sel)), np.mean(np.asarray(covs_naive)))
            print("length so far ", np.mean(np.asarray(lens_sel)), np.mean(np.asarray(lens_naive)))

if __name__ == "__main__":

    test_coverage(nsim=20)