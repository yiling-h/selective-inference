import numpy as np

from selection.randomized.group_lasso import group_lasso
from selection.tests.instance import gaussian_group_instance, gaussian_instance


def test_selected_targets(n=500,
                          p=200,
                          signal_fac=1.5,
                          sgroup=3,
                          s =10,
                          groups=np.arange(50).repeat(4),
                          sigma=2,
                          rho=0.4,
                          randomizer_scale=1,
                          weight_frac=1.5):

    #inst = gaussian_group_instance
    inst = gaussian_instance
    const = group_lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

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
    weights = dict([(i, weight_frac * sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])

    conv = const(X, Y, groups, weights, randomizer_scale=randomizer_scale * sigma_)
    signs = conv.fit()
    nonzero = signs != 0
    print("check dimensions of selected set ", nonzero.sum())

    if nonzero.sum() > 0:
        if n>p:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        estimate, _, _, pval, intervals, _ = conv.selective_MLE(dispersion=dispersion)

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        coverage = (beta_target > intervals[:, 0]) * (beta_target <
                                                      intervals[:, 1])
        return pval[beta[nonzero] == 0], pval[
            beta[nonzero] != 0], coverage, intervals


def main(nsim=500, full=False):
    P0, PA, cover = [], [], []

    n, p, sgroup = 500, 200, 5

    for i in range(nsim):
        if full:
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False
            p0, pA, cover_, intervals = test_full_targets(
                n=n, p=p, s=sgroup, full_dispersion=full_dispersion)
            avg_length = intervals[:, 1] - intervals[:, 0]
        else:
            full_dispersion = True
            p0, pA, cover_, intervals = test_selected_targets(
                n=n, p=p, sgroup=sgroup, full_dispersion=full_dispersion)
            avg_length = intervals[:, 1] - intervals[:, 0]

        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(
            np.array(PA) < 0.1, np.mean(P0), np.std(P0),
            np.mean(np.array(P0) < 0.1), np.mean(np.array(PA) < 0.1),
            np.mean(cover), np.mean(avg_length),
            'null pvalue + power + length')

main(nsim=1, full=False)
