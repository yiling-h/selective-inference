import numpy as np

from selectinf.randomized.group_lasso_MLE import group_lasso, posterior
from selectinf.randomized.tests.instance import gaussian_group_instance


def test_posterior_inference(n=500,
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
                                    groups,
                                    weights,
                                    randomizer_scale= randomizer_scale * sigma_)

        signs, _ = conv.fit()
        nonzero = signs != 0

        conv._setup_implied_gaussian()

        ## define prior for post-selective parameters

        def prior(target_parameter, prior_var=100 * sigma_):
            grad_prior = -target_parameter / prior_var
            log_prior = -np.linalg.norm(target_parameter) ** 2 / (2. * prior_var)
            return grad_prior, log_prior

        if nonzero.sum() > 0:

            ##target of inference: best linear approximation using selected covariates

            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            ##set up selection-informed posterior

            posterior_inf = posterior(conv,
                                      prior=prior,
                                      dispersion=dispersion,
                                      XrawE=False,
                                      useJacobian=True)

            ##run prototype gradient-based sampler

            samples = posterior_inf.langevin_sampler(nsample=1500,
                                                     nburnin=100,
                                                     step=1.)

            ##compute two-sided 90% credible intervals

            alpha = 100 * (1.- level)/2.

            lci = np.percentile(samples, alpha, axis=0)
            uci = np.percentile(samples, 100-alpha, axis=0)
            coverage = (lci < beta_target) * (uci > beta_target)
            length = uci - lci

            return samples, coverage, length

def main(ndraw=10):

    for n in range(ndraw):
        samples, coverage, length = test_posterior_inference(n=500,
                                                             p=200,
                                                             signal_fac=1.2,
                                                             sgroup=3,
                                                             groups=np.arange(50).repeat(4),
                                                             sigma=3.,
                                                             rho=0.40,
                                                             randomizer_scale=0.71,
                                                             weight_frac=1.2,
                                                             level=0.90)

        print("iteration completed ", n + 1, coverage, length)

if __name__ == "__main__":
    main(ndraw=1)
