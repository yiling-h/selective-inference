import numpy as np
from scipy.stats import norm as ndist

from ..multitask_lasso import multi_task_lasso
from ...tests.instance import gaussian_multitask_instance

def test_multitask_lasso_global(ntask=2,
                                nsamples=500 * np.ones(2),
                                p=100,
                                global_sparsity=.8,
                                task_sparsity=.3,
                                sigma=1.*np.ones(2),
                                signal_fac= 0.5,
                                rhos=0.*np.ones(2),
                                weight=2.):

    nsamples = nsamples.astype(int)

    signal = np.sqrt(signal_fac * 2 * np.log(p))

    response_vars, predictor_vars, beta, _gaussian_noise = gaussian_multitask_instance(ntask,
                                                                                       nsamples,
                                                                                       p,
                                                                                       global_sparsity,
                                                                                       task_sparsity,
                                                                                       sigma,
                                                                                       signal,
                                                                                       rhos,
                                                                                       random_signs=False,
                                                                                       equicorrelated=False)[:4]

    feature_weight = weight * np.ones(p)

    #sigmas_ = np.array([np.std(response_vars[i]) for i in range(ntask)])
    sigmas_ = sigma

    randomizer_scales = 0.71 * sigmas_

    #ridge_terms = np.array([np.std(response_vars[i]) * np.sqrt(np.mean((predictor_vars[i] ** 2).sum(0)))/ np.sqrt(nsamples[i] - 1)
    #                          for i in range(ntask)])

    _initial_omega = np.array([randomizer_scales[i] * _gaussian_noise[(i * p):((i + 1) * p)] for i in range(ntask)]).T
   
    multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                            response_vars,
                                            feature_weight,
                                            ridge_term=None,
                                            randomizer_scales=randomizer_scales)
    multi_lasso.fit(perturbations=_initial_omega)

    # dispersions = np.array([np.linalg.norm(response_vars[i] -
    #                                        predictor_vars[i].dot(np.linalg.pinv(predictor_vars[i]).dot(response_vars[i]))) ** 2 / (nsamples[i] - p)
    #                        for i in range(ntask)])

    dispersions = sigma ** 2

    estimate, observed_info_mean, _, _, intervals = multi_lasso.multitask_inference_global(dispersions=dispersions)

    beta_target_ = []

    for j in range(ntask):
        beta_target_.extend(np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(predictor_vars[j].dot(beta[:,j])))

    beta_target_ = np.asarray(beta_target_)
    beta_target = multi_lasso.W_coef.dot(beta_target_)

    coverage = (beta_target > intervals[:, 0]) * (beta_target <
                                                  intervals[:, 1])

    pivot_ = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))
    pivot = 2 * np.minimum(pivot_, 1. - pivot_)

    return coverage, intervals[:, 1] - intervals[:, 0], pivot


def test_multitask_lasso_naive_global(ntask=2,
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

        estimate, observed_info_mean, _, _, intervals = multi_lasso.multitask_inference_global(dispersions=dispersions)

        coverage = []
        pivot = []

        if (active_signs != 0).sum() > 0:

            beta_target_ = []
            observed_target_ = []
            tot_par = multi_lasso.active_global.shape[0]

            prec_target = np.zeros((tot_par, tot_par))
            for j in range(ntask):
                beta_target_.extend(np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(
                    predictor_vars[j].dot(beta[:, j])))
                Qfeat = np.linalg.inv((predictor_vars[j])[:, multi_lasso.active_global].T.dot((predictor_vars[j])[:, multi_lasso.active_global]))
                observed_target_.extend(np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(response_vars[j]))
                prec_target += np.linalg.inv(Qfeat * dispersions[j])

            beta_target_ = np.asarray(beta_target_)
            beta_target = multi_lasso.W_coef.dot(beta_target_)
            observed_target_ = np.asarray(observed_target_)
            observed_target = multi_lasso.W_coef.dot(observed_target_)
            cov_target = np.linalg.inv(prec_target)

            alpha = 1. - 0.90
            quantile = ndist.ppf(1 - alpha / 2.)
            intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                   observed_target + quantile * np.sqrt(np.diag(cov_target))]).T
            coverage.extend((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
            pivot_ = ndist.cdf((observed_target - beta_target) / np.sqrt(np.diag(cov_target)))
            pivot.extend(2 * np.minimum(pivot_, 1. - pivot_))

            return np.asarray(coverage), intervals[:, 1] - intervals[:, 0], np.asarray(pivot)

def test_multitask_lasso_hetero(ntask=2,
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

        if (active_signs != 0).sum()>0:

            dispersions = sigma ** 2

            estimate, observed_info_mean, Z_scores, pvalues, intervals = multi_lasso.multitask_inference_hetero(dispersions=dispersions)

            beta_target = []

            for i in range(ntask):
                X, y = multi_lasso.loglikes[i].data
                beta_target.extend(np.linalg.pinv(X[:, (active_signs[:, i] != 0)]).dot(X.dot(beta[:, i])))

            beta_target = np.asarray(beta_target)
            pivot_ = ndist.cdf((estimate - beta_target)/np.sqrt(np.diag(observed_info_mean)))
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



import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

def test_coverage(nsim=100):

    cov = []
    len = []
    pivots = []

    for n in range(nsim):

        ntask = 5

        coverage, length, pivot = test_multitask_lasso_global(ntask=ntask,
                                                              nsamples=1000 * np.ones(ntask),
                                                              p=50,
                                                              global_sparsity=0.95,
                                                              task_sparsity=0.20,
                                                              sigma=1. * np.ones(ntask),
                                                              signal_fac=0.5,
                                                              rhos=0.50 * np.ones(ntask),
                                                              weight=1.)

        # coverage, length, pivot = test_multitask_lasso_naive_global(ntask=ntask,
        #                                                             nsamples=1000 * np.ones(ntask),
        #                                                             p=50,
        #                                                             global_sparsity=0.95,
        #                                                             task_sparsity=0.20,
        #                                                             sigma=1. * np.ones(ntask),
        #                                                             signal_fac=1.,
        #                                                             rhos=0.50 * np.ones(ntask),
        #                                                             weight=1.)

        # coverage, length, pivot = test_multitask_lasso_hetero(ntask=ntask,
        #                                                       nsamples=1000 * np.ones(ntask),
        #                                                       p=50,
        #                                                       global_sparsity=0.95,
        #                                                       task_sparsity=0.20,
        #                                                       sigma=1. * np.ones(ntask),
        #                                                       signal_fac=np.array([1., 5.]),
        #                                                       rhos=0.50 * np.ones(ntask),
        #                                                       weight=1.)

        # coverage, length, pivot = test_multitask_lasso_naive_hetero(ntask=ntask,
        #                                                             nsamples=1000 * np.ones(ntask),
        #                                                             p=50,
        #                                                             global_sparsity=0.95,
        #                                                             task_sparsity=0.20,
        #                                                             sigma=1. * np.ones(ntask),
        #                                                             signal_fac=np.array([1., 5.]),
        #                                                             rhos=0.50 * np.ones(ntask),
        #                                                             weight=1.)


        cov.extend(coverage)
        len.extend(length)
        pivots.extend(pivot)

        print("iteration completed ", n)
        print("coverage so far ", np.mean(np.asarray(cov)))
        print("length so far ", np.mean(np.asarray(length)))

    plt.clf()
    ecdf_MLE = ECDF(np.asarray(pivots))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

if __name__ == "__main__":

    test_coverage(nsim=20)