from __future__ import print_function
import sys
import os

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_density
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.cisEQTLS.Simes_selection import BH_q


def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma,
                           bh_level,
                           lam_frac = 1.2,
                           loss='gaussian',
                           randomizer='gaussian'):

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
    if randomizer == 'gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer == 'laplace':
        randomization = randomization.laplace((p,), scale=1.)

    M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    print("lasso selects", nactive)

    if nactive == 0:
        return None
    sys.stderr.write("Active set selected by lasso"+str(active_set)+"\n")

    true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
    sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

    ci = approximate_conditional_density(M_est)
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

    p_BH = BH_q(pivots, bh_level)
    discoveries_active = np.zeros(nactive)
    if p_BH is not None:
        for indx in p_BH[1]:
            discoveries_active[indx] = 1

    list_results = np.transpose(np.vstack((sel_covered,
                                           sel_length,
                                           pivots,
                                           naive_covered,
                                           naive_pvals,
                                           naive_length,
                                           active_set,
                                           discoveries_active)))


    print("list of results", list_results)
    return list_results

# n = 350
# p = 5000
# s = 5
# snr = 5.
# bh_level = 0.10
# #
# sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)
# X, y, beta, nonzero, sigma = sample.generate_response()
# results = randomized_lasso_trial(X,
#                                  y,
#                                  beta,
#                                  sigma,
#                                  bh_level,
#                                  lam_frac = 1.2,
#                                  loss='gaussian',
#                                  randomizer='gaussian')
#
# print(results)

if __name__ == "__main__":

#read from command line

    seedn = int(sys.argv[1])
    outdir = sys.argv[2]

    outfile = os.path.join(outdir,"list_result_"+str(seedn)+".txt")

#print("Will save to: "+outfile)

    ### set parameters
    n = 350
    p = 7000
    s = 0
    snr = 5.
    bh_level = 0.20

### GENERATE X
    np.random.seed(0) # ensures same X

    sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)

### GENERATE Y BASED ON SEED
    np.random.seed(seedn) # ensures different y
    X, y, beta, nonzero, sigma = sample.generate_response()

### RUN LASSO AND INFERENCE
    random_lasso = randomized_lasso_trial(X,
                                          y,
                                          beta,
                                          sigma,
                                          bh_level)

### SAVE RESULT
    np.savetxt(outfile, random_lasso)


# def multiple_trials(test_function = randomized_lasso_trial, n = 350, p = 5000, s = 10, snr = 5., bh_level = 0.10, seed_number = 0):
#
#     np.random.seed(seed_number)
#
#     sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)
#
#     adjusted_coverage = 0.
#     unadjusted_coverage = 0.
#
#     adjusted_lengths = 0.
#     unadjusted_lengths = 0.
#
#     FDR = 0.
#     power = 0.
#
#     niter = 10
#     for iter in xrange(niter):
#
#         X, y, beta, nonzero, sigma = sample.generate_response()
#
#         list_results = test_function(X,
#                                      y,
#                                      beta,
#                                      sigma)
#
#         sel_covered = list_results[:,0]
#         sel_length = list_results[:,1]
#         pivots = list_results[:,2]
#         naive_covered = list_results[:,3]
#         naive_pvals = list_results[:,4]
#         naive_length = list_results[:,5]
#         active_set = list_results[:,6]
#
#         nactive = sel_covered.shape[0]
#
#         adjusted_coverage += float(sel_covered.sum() / nactive)
#         unadjusted_coverage += float(naive_covered.sum() / nactive)
#
#         adjusted_lengths += float(sel_length.sum() / nactive)
#         unadjusted_lengths += float(naive_length.sum() / nactive)
#
#         p_BH = BH_q(pivots, bh_level)
#         false_discoveries = 0.
#         true_discoveries = 0.
#
#         if p_BH is not None:
#             for indx in p_BH[1]:
#                 if beta[active_set[indx]] == 0:
#                     false_discoveries += 1.
#                 else:
#                     true_discoveries += 1.
#
#         FDR += false_discoveries / max(float(p_BH[1].shape[0], 1.))
#         power += true_discoveries / float(s)
#
#         print("\n")
#         print("iteration completed", iter + 1)
#         print("results", adjusted_lengths, unadjusted_lengths, FDR, power)
#
#     print(adjusted_coverage, unadjusted_coverage, adjusted_lengths, unadjusted_lengths, FDR, power)
