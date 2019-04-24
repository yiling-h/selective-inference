import numpy as np, sys
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance
from selection.bayesian.utils import projected_langevin, gradient_log_likelihood

def test_approx_pivot(n= 500,
                      p= 100,
                      signal_fac= 1.,
                      s= 5,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)

        print("sigma estimated and true ", sigma, sigma_)

        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = lasso.gaussian(X,
                              y,
                              W,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

        MLE_estimate, _, _, _, _, _ = conv.selective_MLE(observed_target,
                                                         cov_target,
                                                         cov_target_score,
                                                         alternatives)

        def grad_posterior(par):

            log_lik = gradient_log_likelihood(par,
                                              observed_target,
                                              cov_target,
                                              cov_target_score,
                                              conv.observed_opt_state,
                                              conv.cond_mean,
                                              conv.cond_cov,
                                              conv.logdens_linear,
                                              conv.A_scaling,
                                              conv.b_scaling)

            return log_lik

        state = np.zeros(nonzero.sum())
        stepsize = 1. / (0.5 * nonzero.sum())
        sampler = projected_langevin(state, grad_posterior, stepsize)

        samples = []

        for i in range(2000):
            sampler.next()
            samples.append(sampler.state.copy())
            sys.stderr.write("sample number: " + str(i) + "\n")

        samples = np.array(samples)

        print("sample quantiles ", np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0))

        return samples

test_approx_pivot(n= 100,
                  p= 1000,
                  signal_fac= 1.,
                  s= 10,
                  sigma= 1.,
                  rho= 0.40,
                  randomizer_scale= 1.)