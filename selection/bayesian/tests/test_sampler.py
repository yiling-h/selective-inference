import numpy as np
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
#from selection.bayesian.generative_instance import generate_data, generate_data_new
from selection.bayesian.posterior_lasso import inference_lasso

def test_approx_bayesian(n= 500,
                         p= 100,
                         sigma= 1.,
                         rho= 0.40,
                         randomizer_scale= 1.,
                         target ="selected"):

    while True:

        X = np.load("/Users/psnigdha/Research/RadioiBAG/Data/X.npy")
        n, p = X.shape

        X -= X.mean(0)[None, :]
        scalingX = (X.std(0)[None, :] * np.sqrt(n))
        X /= scalingX

        y = np.load("/Users/psnigdha/Research/RadioiBAG/Data/y.npy")
        y = np.log(y)
        y = y - np.mean(y)
        beta = np.zeros(p)

        # beta_true = np.zeros(p)
        # u = np.random.uniform(0., 1., p)
        # for i in range(p):
        #     if u[i] <= 0.80:
        #         beta_true[i] = np.random.laplace(loc=0., scale=0.10)
        #     else:
        #         beta_true[i] = np.random.laplace(loc=0., scale=2.5)
        #
        # beta = beta_true
        #
        # y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

        # X, y, beta, sigma, _ = generate_data_new(n=n, p=p, sigma=sigma, rho=rho, scale =True, center=True)

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)
        print("sigmas ", sigma, sigma_)
        lam_theory = 0.85 * sigma_ * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

        conv = lasso.gaussian(X,
                              y,
                              np.ones(X.shape[1]) * lam_theory,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        if target == "selected":
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

        else:
            beta_target = beta[nonzero]
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = debiased_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              penalty=conv.penalty,
                                              dispersion=dispersion)

        initial_par, _, _, _, _, _ = conv.selective_MLE(observed_target,
                                                        cov_target,
                                                        cov_target_score,
                                                        alternatives)


        posterior_inf = inference_lasso(observed_target,
                                        cov_target,
                                        cov_target_score,
                                        conv.observed_opt_state,
                                        conv.cond_mean,
                                        conv.cond_cov,
                                        conv.logdens_linear,
                                        conv.A_scaling,
                                        conv.b_scaling,
                                        initial_par)

        samples, _  = posterior_inf.posterior_sampler(nsample= 2000, nburnin=50)
        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        print("check target ", lci, beta_target, uci)
        coverage = np.mean((lci < beta_target) * (uci > beta_target))
        length = np.mean(uci - lci)

        return coverage, length


def main(ndraw=10, randomizer_scale=1.):

    coverage_ = 0.
    length_ = 0.
    exception = 0.
    for n in range(ndraw):
        try:
            cov, len = test_approx_bayesian(n=65,
                                            p=300,
                                            sigma=1.,
                                            rho=0.40,
                                            randomizer_scale=randomizer_scale,
                                            target="selected")
        except ValueError:
            exception += 1.
            pass

        coverage_ += cov
        length_ += len

        print("coverage so far ", coverage_ / (n + 1.))
        print("lengths so far ", length_ / (n + 1.))
        print("iteration completed ", n + 1)
        print("exceptions ", exception)

main(ndraw=50)