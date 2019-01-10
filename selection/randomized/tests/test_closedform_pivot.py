import numpy as np
import nose.tools as nt

from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets
from selection.tests.instance import gaussian_instance


def test_full_targets(n=500,
                      p=100,
                      signal_fac=2.0,
                      s=5,
                      sigma=3,
                      rho=0.4,
                      randomizer_scale=1.,
                      full_dispersion=False):


    inst, const = gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2 * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        idx = np.arange(p)
        sigmaX = rho ** np.abs(np.subtract.outer(idx, idx))
        print("snr", beta.T.dot(sigmaX).dot(beta) / ((sigma ** 2.) * n))

        n, p = X.shape

        sigma_ = np.std(Y)
        W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_

        conv = const(X,
                     Y,
                     W,
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        print("dimensions", n, p, nonzero.sum())

        if nonzero.sum() > 0:
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = None

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = full_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)

            intervals, pvalue = conv.inference_new(observed_target,
                                                   cov_target,
                                                   cov_target_score,
                                                   alternatives)

            MLE_intervals = conv.selective_MLE(observed_target,
                                               cov_target,
                                               cov_target_score,
                                               alternatives)[4]

            print("intervals", intervals.T, MLE_intervals)
            break

test_full_targets()