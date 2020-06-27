import numpy as np

from selection.randomized.group_lasso import group_lasso
from selection.tests.instance import gaussian_group_instance, gaussian_instance

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm as ndist

from selection.randomized.lasso import lasso, selected_targets

def test_selected_targets(n=500,
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
    const = group_lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    # X, Y, beta = inst(n=n,
    #                   p=p,
    #                   signal=signal,
    #                   s=s,
    #                   equicorrelated=False,
    #                   rho=rho,
    #                   sigma=sigma,
    #                   random_signs=True)[:3]

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

    sigma_ = np.std(Y)
    if n > p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)
    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])
    conv = const(X, Y, groups, weights, randomizer_scale=randomizer_scale * sigma_)
    signs, _ = conv.fit()
    nonzero = signs != 0
    print("check dimensions of selected set ", nonzero.sum())

    if nonzero.sum() > 0:
        if n > p:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        estimate, observed_info_mean, _, pval, intervals, _ = conv.selective_MLE(dispersion=dispersion)

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        pivot_MLE = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(observed_info_mean)))

        coverage = (beta_target > intervals[:, 0]) * (beta_target <
                                                      intervals[:, 1])

        naive_estimate = np.linalg.pinv(X[:, nonzero]).dot(Y)

        naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, nonzero].T.dot(X[:, nonzero])))))
        naive_intervals = np.vstack([naive_estimate - ndist.ppf(0.95) * naive_sd,
                                     naive_estimate + ndist.ppf(0.95) * naive_sd]).T
        coverage_naive = (beta_target > naive_intervals[:, 0]) * (beta_target <
                                                      naive_intervals[:, 1])

        return pval[beta[nonzero] == 0], pval[
            beta[nonzero] != 0], coverage, intervals, pivot_MLE, coverage_naive


def main(nsim=500):
    P0, PA, cover, pivot, cover_naive = [], [], [], [], []

    n, p, sgroup = 500, 200, 3

    for i in range(nsim):
        p0, pA, cover_, intervals, pivot_, cover_naive_ = test_selected_targets(n=n, p=p, sgroup=sgroup)
        avg_length = intervals[:, 1] - intervals[:, 0]

        cover.extend(cover_)
        cover_naive.extend(cover_naive_)
        pivot.extend(pivot_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover), np.mean(avg_length),
            'coverage + length so far')
        print(np.mean(cover_naive), 'coverage naive so far')

    plt.clf()
    ecdf_MLE = ECDF(np.asarray(pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()

if __name__ == "__main__":

    main(nsim=500)
