from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_density
from selection.frequentist_eQTL.estimator import M_estimator_2step

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage
from selection.frequentist_eQTL.simes_BH_selection import simes_selection, BH_simes, BH_selection_egenes, simes_selection_egenes

from selection.api import randomization

n = 350
p = 1000
s = 5
snr = 5.
bh_level = 0.10

sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)
X, y, beta, nonzero, sigma = sample.generate_response()
lam_frac = 1.
lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
loss = rr.glm.gaussian(X, y)
epsilon = 1. / np.sqrt(n)
n, p = X.shape

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p), weights=dict(zip(np.arange(p), W)), lagrange=1.)
randomization = randomization.isotropic_gaussian((p,), scale=1.)

simes = simes_selection_egenes(X, y)
info = simes.post_BH_selection(0.05)

index = info[2]
J = info[1]
T_sign = info[3]
i_0 = info[0]
threshold = np.zeros(i_0+1)
simes_level = 0.05

if i_0>0:
    threshold[0] = np.sqrt(2.)* normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
    threshold[1:] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (np.arange(i_0)+1))
    data_simes = np.zeros(i_0+1)
    data_simes[0] = X[:, index].T.dot(y)
    data_simes[1:] = X[:, J].T.dot(y)
else:
    threshold[0] = np.sqrt(2.)* normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
    data_simes = X[:, index].T.dot(y)

M_est = M_estimator_2step(loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold, data_simes)
M_est.solve_approx()
active = M_est._overall
active_set = np.asarray([i for i in range(p) if active[i]])
nactive = np.sum(active)

M_est.setup_map(1)

prob = approximate_conditional_prob_2stage(0.5, M_est)

sol = prob.minimize2(nstep=100)
print(-sol[1])