from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.estimator import M_estimator_exact, M_estimator_2step
from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage, approximate_conditional_density_2stage

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

#from selection.bayesian.initial_soln import instance
from selection.frequentist_eQTL.simes_BH_selection import BH_selection_egenes, simes_selection_egenes
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.frequentist_eQTL.instance import instance


def random_lasso(X,
                 y,
                 beta,
                 sigma,
                 s,
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

    M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer='gaussian')
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])

    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")

    power = active[:s].sum()

    return power

if __name__ == "__main__":
    n = 350
    p = 1000
    s = 3

    np.random.seed(0)
    sample = instance(n=n, p=p, s=s, sigma=1., rho=0)
    power = 0.

    for i in range(100):
        np.random.seed(i)  # ensures different y for the same X
        X, y, beta, nonzero, sigma = sample.generate_response()
        power += random_lasso(X,
                              y,
                              beta,
                              sigma,
                              s=3)

    print(power/3)

