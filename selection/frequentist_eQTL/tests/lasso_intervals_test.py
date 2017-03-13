from __future__ import print_function
import numpy as np
import regreg.api as rr

from selection.tests.instance import logistic_instance, gaussian_instance
from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_density
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

def test_lasso(n=350,
               p=5000,
               s=10,
               snr=5.,
               rho=0.,
               lam_frac = 1.2,
               loss='gaussian',
               randomizer='gaussian'):

    from selection.api import randomization
    if snr == 0:
        s = 0
    if loss == "gaussian":
        X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, rho=rho, snr=snr, sigma=1.)
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    true_support = nonzero
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
    print("active set, true_support", active_set, true_support)
    true_vec = np.linalg.inv(X[:,active].T.dot(X[:,active])).dot(X[:,active].T).dot(X.dot(beta))
    print("true target", true_vec)

    #if (set(active_set).intersection(set(true_support)) == set(true_support)) == True:

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
    naive_covered = np.zeros(nactive)
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

    return sel_covered, sel_length, pivots, naive_covered, naive_pvals, naive_length


print(test_lasso())