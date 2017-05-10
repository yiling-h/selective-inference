from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.estimator import M_estimator_2step
from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage, approximate_conditional_density_2stage

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import instance
from selection.frequentist_eQTL.simes_BH_selection import BH_selection_egenes, simes_selection_egenes
from selection.bayesian.cisEQTLS.Simes_selection import BH_q


def hierarchical_lasso_trial(X,
                             y,
                             beta,
                             sigma,
                             simes_level,
                             index,
                             J,
                             T_sign,
                             threshold,
                             data_simes,
                             bh_level = 0.10,
                             lam_frac=1.,
                             loss='gaussian'):

    from selection.api import randomization
    if beta[0] == 0:
        s = 0

    n, p = X.shape
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    M_est = M_estimator_2step(loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold,
                              data_simes)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")

    if nactive == 0:
        return None

    else:
        true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

        ci = approximate_conditional_density_2stage(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_covered = np.zeros(nactive, np.bool)
        sel_length = np.zeros(nactive)
        pivots = np.zeros(nactive)

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = np.zeros(nactive)

        for j in xrange(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1
            sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
            print(ci_sel[j, :])
            pivots[j] = ci.approximate_pvalue(j, 0.)

            # naive ci
            if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                naive_covered[j] += 1
            naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]

        sys.stderr.write("Total adjusted covered" + str(sel_covered.sum()) + "\n")
        sys.stderr.write("Total naive covered" + str(naive_covered.sum()) + "\n")

        power = 0.
        false_discoveries = 0.
        beta_active = beta[active]
        p_BH = BH_q(pivots, bh_level)
        discoveries_active = np.zeros(nactive)
        if p_BH is not None:
            for indx in p_BH[1]:
                discoveries_active[indx] = 1
                if beta_active[indx] != 0:
                    power += 1.
                else:
                    false_discoveries += 1.

        power = power/5.
        fdr = false_discoveries/(max(1.,discoveries_active.sum()))

        sys.stderr.write("Power" + str(power) + "\n")
        sys.stderr.write("FDR" + str(fdr) + "\n")

        list_results = np.transpose(np.vstack((sel_covered,
                                               sel_length,
                                               pivots,
                                               naive_covered,
                                               naive_pvals,
                                               naive_length,
                                               active_set,
                                               discoveries_active)))


        return list_results



if __name__ == "__main__":
    ### set parameters
    n = 350
    p = 1000
    s = 0
    snr = 5.
    bh_level = 0.20

    egenes =3000
    ngenes = 5000
    simes_level = 0.6*0.20

    np.random.seed(0)  # ensures same X
    sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)

    np.random.seed(3) #ensures different y for the same X
    X, y, beta, nonzero, sigma = sample.generate_response()

    simes = simes_selection_egenes(X, y)
    simes_p = simes.simes_p_value()
    print("simes_p_value", simes_p)
    print("simes level", simes_level)

    #proceed only if gene is an eGene, that is simes p-value is significant at the  BH cut-off level
    if simes_p <= simes_level:

        sig_simes = simes.post_BH_selection(simes_level)
        index = sig_simes[2]
        J = sig_simes[1]
        T_sign = sig_simes[3]
        i_0 = sig_simes[0]
        threshold = np.zeros(i_0 + 1)


        ###setting the parameters from Simes
        if i_0 > 0:
            threshold[0] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
            threshold[1:] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (np.arange(i_0) + 1))
            data_simes = np.zeros(i_0 + 1)
            data_simes[0] = X[:, index].T.dot(y)
            data_simes[1:] = X[:, J].T.dot(y)
        else:
            threshold[0] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
            data_simes = X[:, index].T.dot(y)

        results = hierarchical_lasso_trial(X,
                                           y,
                                           beta,
                                           sigma,
                                           simes_level,
                                           index,
                                           J,
                                           T_sign,
                                           threshold,
                                           data_simes,
                                           bh_level=0.10,
                                           lam_frac=1.,
                                           loss='gaussian')

        print("inferential output", results)


