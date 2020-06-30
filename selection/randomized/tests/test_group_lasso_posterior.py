import numpy as np

from selection.randomized.group_lasso import group_lasso, posterior
from selection.tests.instance import gaussian_group_instance, gaussian_instance


def test_posterior_inference(n=500,
                             p=200,
                             signal_fac=0.1,
                             sgroup=3,
                             s =5,
                             groups=np.arange(50).repeat(4),
                             sigma=3.,
                             rho=0.3,
                             randomizer_scale=1,
                             weight_frac=1.2):

    inst = gaussian_group_instance
    #inst = gaussian_instance
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

    # X, Y, beta = inst(n=n,
    #                   p=p,
    #                   signal=signal,
    #                   s=s,
    #                   equicorrelated=False,
    #                   rho=rho,
    #                   sigma=sigma,
    #                   random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)
    print("check dispersion ", dispersion)

    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
    conv = group_lasso.gaussian(X,
                                Y,
                                groups,
                                weights,
                                randomizer_scale=randomizer_scale * sigma_)

    signs, _ = conv.fit()
    nonzero = signs != 0
    print("dimensions ", nonzero.sum())

    conv._setup_implied_gaussian()

    def prior(target_parameter):
        grad_prior = -target_parameter / 100
        log_prior = -np.linalg.norm(target_parameter) ** 2 / (2. * 100)
        return grad_prior, log_prior

    if nonzero.sum() > 0:
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        posterior_inf = posterior(conv,
                                  prior=prior,
                                  dispersion=dispersion)

        samples = posterior_inf.langevin_sampler(nsample=1200,
                                                 nburnin=100,
                                                 step=1.)

        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        return np.mean(coverage), np.mean(length)



def main(ndraw=10):

    coverage_ = 0.
    length_ = 0.
    for n in range(ndraw):
        cov, len = test_posterior_inference(n=500,
                                            p=200,
                                            signal_fac=0.1,
                                            sgroup=3,
                                            s=5,
                                            groups=np.arange(50).repeat(4),
                                            sigma=3.,
                                            rho=0.20,
                                            randomizer_scale=1.,
                                            weight_frac=1.2)

        coverage_ += cov
        length_ += len

        print("coverage so far ", coverage_ / (n + 1.))
        print("lengths so far ", length_ / (n + 1.))
        print("iteration completed ", n + 1)


if __name__ == "__main__":
    main()