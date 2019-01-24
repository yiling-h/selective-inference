import numpy as np
from scipy.stats import norm as ndist

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

from selection.tests.instance import gaussian_instance
from selection.tests.decorators import rpy_test_safe

from selection.randomized.screening import stepup, stepup_selection, marginal_screening
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

def coverage(intervals, pval, target, truth, alpha):
    if pval is not None:
        pval_ralt = (pval[truth != 0]) < alpha
        pval_rnull = (pval[truth == 0]) < alpha
        if pval_ralt.sum() > 0:
            avg_power = np.mean(pval_ralt)
        else:
            avg_power = 0.
        fdr = pval_rnull.sum()/float(max((pval<alpha).sum(), 1.))

        return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power, fdr, (pval<alpha).sum()
    else:
        intervals_nonnull = intervals[(truth != 0), :]
        intervals_null = intervals[(truth == 0), :]
        tot_reported = np.array(1-((intervals[:, 0]<=0.)*(intervals[:, 1]>=0.)), np.bool).sum()
        intervals_ralt = np.array(1-((intervals_nonnull[:, 0]<=0.)*(intervals_nonnull[:, 1]>=0.)), np.bool)
        intervals_rnull = np.array(1-((intervals_null[:, 0]<=0.)*(intervals_null[:, 1]>=0.)), np.bool)

        return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), np.mean(intervals_ralt), intervals_rnull.sum()/float(max(tot_reported, 1.)), tot_reported

def test_BH(p=500,
            s=50,
            sigma = 3.,
            rho=0.35,
            randomizer_scale=1.,
            target = "full",
            level=0.9,
            q=0.1):

    W = rho ** (np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
    sqrtW = np.linalg.cholesky(W)
    Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
    beta = np.zeros(p)
    beta[:s] = (2 * np.random.binomial(1, 0.5, size=(s,)) - 1) * np.linspace(3.5, 3.5, s) * sigma
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
                                  useC=True)

    nonrand_nonzero = nonrand_BH_select.fit()

    BH_select = stepup.BH(score,
                          W * sigma ** 2,
                          randomizer_scale * sigma,
                          q=q,
                          perturb=omega,
                          useC=True)

    nonzero = BH_select.fit()
    print("selected", nonzero.sum(), nonrand_nonzero.sum())

    naive_nreport, sel_nreport = [0.,0.]

    if target == "marginal":
        beta_target = true_mean[nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = BH_select.marginal_targets(nonzero)

        nonrand_beta_target = true_mean[nonrand_nonzero]
        (nonrand_observed_target,
         nonrand_cov_target,
         nonrand_crosscov_target_score,
         nonrand_alternatives) = nonrand_BH_select.marginal_targets(nonrand_nonzero)
    else:
        beta_target = beta[nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = BH_select.full_targets(nonzero, dispersion=sigma ** 2)

        nonrand_beta_target = beta[nonrand_nonzero]
        (nonrand_observed_target,
         nonrand_cov_target,
         nonrand_crosscov_target_score,
         nonrand_alternatives) = nonrand_BH_select.full_targets(nonrand_nonzero)

    alpha = 1. - level
    quantile = ndist.ppf(1 - alpha / 2.)
    adjusted_alpha = (nonrand_nonzero.sum() * alpha) / p
    adjusted_quantile = ndist.ppf(1 - adjusted_alpha / 2.)

    if nonzero.sum()>0:
        sel_estimate, sel_info, _, sel_pval, sel_intervals, _ = BH_select.selective_MLE(observed_target,
                                                                                        cov_target,
                                                                                        crosscov_target_score,
                                                                                        level=level)

        sel_cov, sel_power, sel_fdr, sel_dis = coverage(sel_intervals, sel_pval, beta_target, beta[nonzero], alpha)
        sel_length = np.mean(sel_intervals[:, 1] - sel_intervals[:, 0])
        sel_bias = np.mean(sel_estimate - beta_target)

    else:
        sel_nreport = 1.
        sel_cov, sel_length, sel_bias, sel_power, sel_fdr, sel_dis = [0., 0., 0., 0., 0., 0.]

    if nonrand_nonzero.sum()>0:

        naive_intervals = np.vstack([nonrand_observed_target - quantile * np.sqrt(np.diag(nonrand_cov_target)),
                                     nonrand_observed_target + quantile * np.sqrt(np.diag(nonrand_cov_target))]).T
        naive_pval = 2 * ndist.cdf(np.abs(nonrand_observed_target) / np.sqrt(np.diag(nonrand_cov_target)))
        naive_cov, naive_power, naive_fdr, naive_dis = coverage(naive_intervals, naive_pval, nonrand_beta_target, beta[nonrand_nonzero], alpha)
        naive_length = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
        naive_bias = np.mean(nonrand_observed_target - nonrand_beta_target)

        fcr_intervals = np.vstack([nonrand_observed_target - adjusted_quantile * np.sqrt(np.diag(nonrand_cov_target)),
                                   nonrand_observed_target + adjusted_quantile * np.sqrt(np.diag(nonrand_cov_target))]).T
        fcr_cov, fcr_power, fcr_fdr, fcr_dis = coverage(naive_intervals, None, nonrand_beta_target, beta[nonrand_nonzero], alpha)
        fcr_length = np.mean(fcr_intervals[:, 1] - fcr_intervals[:, 0])

    else:
        naive_nreport = 1.
        naive_cov, naive_length, naive_bias, naive_power, naive_fdr, naive_dis = [0., 0., 0., 0., 0., 0.]
        fcr_cov, fcr_length, fcr_bias, fcr_power, fcr_fdr, fcr_dis = [0., 0., 0., 0., 0., 0.]

    MLE_inf = np.vstack((sel_cov, sel_length, nonzero.sum(), sel_bias, sel_power, sel_fdr, sel_dis))
    naive_inf = np.vstack((naive_cov, naive_length, nonrand_nonzero.sum(), naive_bias, naive_power, naive_fdr, naive_dis))
    fcr_inf = np.vstack((fcr_cov, fcr_length, nonrand_nonzero.sum(), naive_bias, fcr_power, fcr_fdr, fcr_dis))
    nreport = np.vstack((sel_nreport, naive_nreport))

    return np.vstack((MLE_inf, naive_inf, fcr_inf, nreport))

#test_BH()


def test_marginal(p=500,
                  s=50,
                  sigma = 3.,
                  rho=0.35,
                  randomizer_scale=1.,
                  target = "full",
                  level=0.9,
                  q=0.1):

    W = rho ** (np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
    sqrtW = np.linalg.cholesky(W)
    Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
    beta = np.zeros(p)
    beta[:s] = (2 * np.random.binomial(1, 0.5, size=(s,)) - 1) * np.linspace(3.5, 3.5, s) * sigma
    np.random.shuffle(beta)
    true_mean = W.dot(beta)

    score = Z + true_mean
    omega = randomization.isotropic_gaussian((p,), randomizer_scale * sigma).sample()

    nonrand_marginal_select = marginal_screening.type1(score,
                                                       W * sigma ** 2,
                                                       q,
                                                       randomizer_scale * sigma,
                                                       useC=False,
                                                       perturb=np.zeros(p))
    nonrand_boundary = nonrand_marginal_select.fit()
    nonrand_nonzero = nonrand_boundary != 0

    marginal_select = marginal_screening.type1(score,
                                               W * sigma ** 2,
                                               q,
                                               randomizer_scale * sigma,
                                               useC=False,
                                               perturb=omega)

    boundary = marginal_select.fit()
    nonzero = boundary != 0

    print("selected", nonzero.sum(), nonrand_nonzero.sum())

    naive_nreport, sel_nreport = [0.,0.]

    if target == "marginal":
        beta_target = true_mean[nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = marginal_select.marginal_targets(nonzero)

        nonrand_beta_target = true_mean[nonrand_nonzero]
        (nonrand_observed_target,
         nonrand_cov_target,
         nonrand_crosscov_target_score,
         nonrand_alternatives) = nonrand_marginal_select.marginal_targets(nonrand_nonzero)
    else:
        beta_target = beta[nonzero]
        (observed_target,
         cov_target,
         crosscov_target_score,
         alternatives) = marginal_select.full_targets(nonzero, dispersion=sigma ** 2)

        nonrand_beta_target = beta[nonrand_nonzero]
        (nonrand_observed_target,
         nonrand_cov_target,
         nonrand_crosscov_target_score,
         nonrand_alternatives) = nonrand_marginal_select.full_targets(nonrand_nonzero)

    alpha = 1. - level
    quantile = ndist.ppf(1 - alpha / 2.)
    adjusted_alpha = (nonrand_nonzero.sum() * alpha) / float(p)
    adjusted_quantile = ndist.ppf(1 - adjusted_alpha /2.)

    if nonzero.sum()>0:
        sel_estimate, sel_info, _, sel_pval, sel_intervals, _ = marginal_select.selective_MLE(observed_target,
                                                                                              cov_target,
                                                                                              crosscov_target_score,
                                                                                              level=level)

        sel_cov, sel_power, sel_fdr, sel_dis = coverage(sel_intervals, sel_pval, beta_target, beta[nonzero], alpha)
        sel_length = np.mean(sel_intervals[:, 1] - sel_intervals[:, 0])
        sel_bias = np.mean(sel_estimate - beta_target)

    else:
        sel_nreport = 1.
        sel_cov, sel_length, sel_bias, sel_power, sel_fdr, sel_dis = [0., 0., 0., 0., 0., 0.]

    if nonrand_nonzero.sum()>0:

        naive_intervals = np.vstack([nonrand_observed_target - quantile * np.sqrt(np.diag(nonrand_cov_target)),
                                     nonrand_observed_target + quantile * np.sqrt(np.diag(nonrand_cov_target))]).T
        naive_pval = 2 * ndist.cdf(np.abs(nonrand_observed_target) / np.sqrt(np.diag(nonrand_cov_target)))
        naive_cov, naive_power, naive_fdr, naive_dis = coverage(naive_intervals, naive_pval, nonrand_beta_target, beta[nonrand_nonzero], alpha)
        naive_length = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
        naive_bias = np.mean(nonrand_observed_target - nonrand_beta_target)

        fcr_intervals = np.vstack([nonrand_observed_target - adjusted_quantile * np.sqrt(np.diag(nonrand_cov_target)),
                                   nonrand_observed_target + adjusted_quantile * np.sqrt(np.diag(nonrand_cov_target))]).T
        fcr_cov, fcr_power, fcr_fdr, fcr_dis = coverage(naive_intervals, None, nonrand_beta_target, beta[nonrand_nonzero], alpha)
        fcr_length = np.mean(fcr_intervals[:, 1] - fcr_intervals[:, 0])

    else:
        naive_nreport = 1.
        naive_cov, naive_length, naive_bias, naive_power, naive_fdr, naive_dis = [0., 0., 0., 0., 0., 0.]
        fcr_cov, fcr_length, fcr_bias, fcr_power, fcr_fdr, fcr_dis = [0., 0., 0., 0., 0., 0.]

    MLE_inf = np.vstack((sel_cov, sel_length, nonzero.sum(), sel_bias, sel_power, sel_fdr, sel_dis))
    naive_inf = np.vstack((naive_cov, naive_length, nonrand_nonzero.sum(), naive_bias, naive_power, naive_fdr, naive_dis))
    fcr_inf = np.vstack((fcr_cov, fcr_length, nonrand_nonzero.sum(), naive_bias, fcr_power, fcr_fdr, fcr_dis))
    nreport = np.vstack((sel_nreport, naive_nreport))

    return np.vstack((MLE_inf, naive_inf, fcr_inf, nreport))

test_marginal()