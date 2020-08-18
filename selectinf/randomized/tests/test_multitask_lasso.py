import numpy as np
from scipy.stats import norm as ndist

from ..multitask_lasso import multi_task_lasso
from ...tests.instance import gaussian_multitask_instance

def main():

    K = 4
    sample_sizes = (200, 200, 200, 200)
    p = 10
    beta = np.random.random((p, K))

    global_sparsity_rate = .90
    task_sparsity_rate = .50
    global_zeros = np.random.choice(p,int(round(global_sparsity_rate*p)))

    beta[global_zeros,:] = np.zeros((K,))
    for i in np.delete(range(p),global_zeros):
        beta[i,np.random.choice(K,int(round(task_sparsity_rate * K)))] = 0.
    print("beta ", beta)

    predictor_vars = {i: np.random.random((sample_sizes[i], p)) for i in range(K)}
    response_vars = {i: np.dot(predictor_vars[i], beta[:, i]) for i in range(K)}
    feature_weight = 1.25 * np.ones(p)
    randomizer_scales = np.ones(K)

    multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                            response_vars,
                                            feature_weight,
                                            randomizer_scales = randomizer_scales)

    print(multi_lasso.fit())


def test_multitask_lasso(ntask=2,
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

    response_vars, predictor_vars, beta = gaussian_multitask_instance(ntask,
                                                                      nsamples,
                                                                      p,
                                                                      global_sparsity,
                                                                      task_sparsity,
                                                                      sigma,
                                                                      signal,
                                                                      rhos)[:3]

    feature_weight = weight * 1. * np.sqrt(2 * np.log(p)) * np.ones(p)

    sigmas_ = np.array([np.std(response_vars[i]) for i in range(ntask)])

    randomizer_scales = 0.71 * sigmas_

    #ridge_terms = np.array([np.std(response_vars[i]) * np.sqrt(np.mean((predictor_vars[i] ** 2).sum(0)))/ np.sqrt(nsamples[i] - 1)
    #                          for i in range(ntask)])
   
    multi_lasso = multi_task_lasso.gaussian(predictor_vars,
                                            response_vars,
                                            feature_weight,
                                            ridge_term = None,
                                            randomizer_scales = randomizer_scales)
    multi_lasso.fit()

    # dispersions = np.array([np.linalg.norm(response_vars[i] -
    #                                        predictor_vars[i].dot(np.linalg.pinv(predictor_vars[i]).dot(response_vars[i]))) ** 2 / (nsamples[i] - p)
    #                        for i in range(ntask)])

    dispersions = sigma

    estimate, observed_info_mean, _, _, intervals = multi_lasso.multitask_inference(dispersions=dispersions)

    beta_target_ = []

    for j in range(ntask):
        beta_target_.extend(np.linalg.pinv((predictor_vars[j])[:, multi_lasso.active_global]).dot(predictor_vars[j].dot(beta[:,j])))

    beta_target_ = np.asarray(beta_target_)
    beta_target = multi_lasso.W_coef.dot(beta_target_)

    print("check size range ", np.amax(np.fabs(beta_target)), np.amin(np.fabs(beta_target)))

    coverage = (beta_target > intervals[:, 0]) * (beta_target <
                                                  intervals[:, 1])

    print("coverage ", coverage)
    if coverage.sum() <= 0.10:
        print(intervals, multi_lasso.observed_opt_states, estimate)
    return coverage, intervals[:, 1]- intervals[:, 0]

def test_coverage(nsim=100):

    cov = []
    len = []

    for n in range(nsim):

        coverage, length = test_multitask_lasso(ntask=3,
                                                nsamples=500 * np.ones(3),
                                                p=50,
                                                global_sparsity=.95,
                                                task_sparsity=0.20,
                                                sigma=1.*np.ones(3),
                                                signal_fac=1.,
                                                rhos=0.50*np.ones(3),
                                                weight=1.)

        if coverage.sum()<= 0.10:
            break

        cov.extend(coverage)
        len.extend(length)

        print("iteration completed ", n)
        print("coverage so far ", np.mean(np.asarray(cov)))
        print("length so far ", np.mean(np.asarray(length)))

if __name__ == "__main__":
    test_coverage(nsim=10)