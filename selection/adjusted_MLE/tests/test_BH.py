import numpy as np
from scipy.stats import norm as ndist

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

from selection.tests.instance import gaussian_instance
from selection.tests.decorators import rpy_test_safe

from selection.randomized.screening import stepup, stepup_selection
from selection.randomized.randomization import randomization

def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)

def coverage(intervals, pval, target, truth):
    pval_alt = (pval[truth != 0]) < 0.1
    if pval_alt.sum() > 0:
        avg_power = np.mean(pval_alt)
    else:
        avg_power = 0.
    return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power

def test_BH(p=500,
            s=50,
            sigma=3.,
            rho=0.35,
            randomizer_scale=1.,
            target = "full",
            level=0.9,
            q=0.1):

    W = rho ** (np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
    sqrtW = np.linalg.cholesky(W)
    Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
    beta = np.zeros(p)
    beta[:s] = (2 * np.random.binomial(1, 0.5, size=(s,)) - 1) * np.linspace(5, 5, s) * sigma
    np.random.shuffle(beta)
    true_mean = W.dot(beta)

    score = Z + true_mean
    omega = randomization.isotropic_gaussian((p,), randomizer_scale * sigma).sample()

    ##using python solver for now:
    nonrand_BH_select = stepup.BH(score,
                                  W * sigma ** 2,
                                  randomizer_scale * sigma,
                                  q=q,
                                  perturb=np.zeros(p),
                                  useC=False)

    nonrand_nonzero = nonrand_BH_select.fit()

    BH_select = stepup.BH(score,
                          W * sigma ** 2,
                          randomizer_scale * sigma,
                          q=q,
                          perturb=omega,
                          useC=False)

    nonzero = BH_select.fit()

    if target == "marginal":
        beta_target = true_mean[nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = BH_select.marginal_targets(nonzero)

        nonrand_beta_target = true_mean[nonrand_nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = nonrand_BH_select.marginal_targets(nonrand_nonzero)
    else:
        beta_target = beta[nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = BH_select.full_targets(nonzero, dispersion=sigma ** 2)

        nonrand_beta_target = beta[nonrand_nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = nonrand_BH_select.full_targets(nonrand_nonzero)

    alpha = 1 - level
    quantile = ndist.ppf(1 - alpha / 2.)

    if nonzero>0:

        sel_estimate, sel_info, _, sel_pval, sel_intervals, _ = BH_select.selective_MLE(observed_target,
                                                                                        cov_target,
                                                                                        crosscov_target_score,
                                                                                        level=level)

        sel_cov, sel_power = coverage(sel_intervals, sel_pval, beta_target, beta[nonzero])
        sel_length = np.mean(sel_intervals[:, 1] - sel_intervals[:, 0])

    if nonrand_nonzero>0:

        naive_intervals = np.vstack([observed_target - quantile * np.sqrt(np.diag(cov_target)),
                                     observed_target + quantile * np.sqrt(np.diag(cov_target))])
        naive_pval = 2 * ndist.cdf(np.abs(observed_target) / np.sqrt(np.diag(cov_target)))
        naive_cov, naive_power = coverage(naive_intervals, naive_pval, beta_target, beta[nonzero])
        naive_length = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])















