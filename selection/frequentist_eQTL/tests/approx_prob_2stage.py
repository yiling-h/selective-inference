from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_density
from selection.frequentist_eQTL.estimator import M_estimator_exact, M_estimator_2step

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage
from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_prob
from selection.frequentist_eQTL.simes_BH_selection import simes_selection, BH_simes, BH_selection_egenes, simes_selection_egenes

from selection.api import randomization

n = 350
p = 1000
s = 0
snr = 5.
bh_level = 0.10
simes_level = 0.6*0.20

np.random.seed(0)  # ensures same X
sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)

np.random.seed(3) #ensures different y for the same X
X, y, beta, nonzero, sigma = sample.generate_response()

simes = simes_selection_egenes(X, y)
simes_p = simes.simes_p_value()
print("simes_p_value", simes_p)
print("simes level", simes_level)

if simes_p <= simes_level:

    sig_simes = simes.post_BH_selection(simes_level)
    index = sig_simes[2]
    print("index", index)
    J = sig_simes[1]
    T_sign = sig_simes[3]
    i_0 = sig_simes[0]
    print("i_0", i_0)
    threshold = np.zeros(i_0 + 1)

    if i_0>0:
        print("not Bonferroni")
        threshold[0] = np.sqrt(2.)* normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
        threshold[1:] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (np.arange(i_0)+1))
        data_simes = np.zeros(i_0+1)
        data_simes[0] = X[:, index].T.dot(y)
        data_simes[1:] = X[:, J].T.dot(y)
    else:
        print("Bonferroni")
        threshold[0] = np.sqrt(2.)* normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
        data_simes = X[:, index].T.dot(y)

    lam_frac = 1.
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
    loss = rr.glm.gaussian(X, y)
    epsilon = 1. / np.sqrt(n)
    n, p = X.shape
    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    M_est = M_estimator_2step(loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold,
                              data_simes)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")

    target_observed = M_est.target_observed
    grid_length = 601
    grid = np.zeros((nactive, grid_length))
    grid[1, :] = np.linspace(target_observed[1] - 15., target_observed[1] + 15., num=601)
    #print("grid", grid[1,:])

    obs = target_observed[1]
    print("target", obs)

    M_est.setup_map(1)

    test = approximate_conditional_prob_2stage((grid[1, :])[300], M_est)

    feasible_point = np.ones(nactive + 1)
    feasible_point[1:] = M_est.feasible_point_lasso
    u = feasible_point
    print("feasible", feasible_point)

    test_objective = test.sel_prob_smooth_objective(u, 'grad')
    print("test_objective", test_objective)


    #########comparison with lasso
    M_est_1 = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer='gaussian')
    M_est_1.solve_approx()
    active_1 = M_est_1._overall
    active_set_1 = np.asarray([i for i in range(p) if active_1[i]])
    nactive_1 = np.sum(active_1)

    obs_1 = M_est_1.target_observed[1]
    print("target", obs_1)

    M_est_1.setup_map(1)

    test_1 = approximate_conditional_prob((grid[1, :])[300], M_est_1)

    feasible_point_1 = M_est_1.feasible_point
    u = feasible_point_1
    print("feasible lasso", feasible_point_1)

    test_objective_1 = test_1.sel_prob_smooth_objective(u, 'grad')
    print("test_objective lasso", test_objective_1)


    # test_prob = (test.minimize2(nstep=100)[::-1])[0]
    # print("test_prob", test_prob)

    # h_hat = []
    #
    # for i in xrange(grid[1, :].shape[0]):
    #     approx = approximate_conditional_prob_2stage((grid[1, :])[i], M_est)
    #     h_hat.append(-(approx.minimize2(nstep=100)[::-1])[0])
    #
    # print(h_hat)