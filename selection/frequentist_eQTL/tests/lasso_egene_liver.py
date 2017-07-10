from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np, statsmodels.api as sm
import regreg.api as rr
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues
from selection.tests.instance import gaussian_instance
from selection.api import randomization
from selection.bayesian.initial_soln import selection

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def estimate_sigma(X, y, nstep=30, tol=1.e-4):

    old_sigma = 0.5
    old_old_sigma = old_sigma
    n, p = X.shape

    for itercount in range(nstep):

        random_Z = np.zeros(p)
        sel = selection(X, y, random_Z, sigma=old_sigma)
        if sel is not None:
            lam, epsilon, active, betaE, cube, initial_soln = sel
            sys.stderr.write("active" + str(active.sum()) + "\n")
            if active.sum()<n-1:
                ols_fit = sm.OLS(y, X[:, active]).fit()
                new_sigma = np.linalg.norm(ols_fit.resid) / np.sqrt(n - active.sum() - 1)
            else:
                new_sigma = 0.75
        else:
            new_sigma = old_sigma/2.

        sys.stderr.write("est_sigma" + str(new_sigma) + str(old_sigma)+ "\n")
        if np.fabs(new_sigma - old_sigma) < tol :
            sigma = new_sigma
            break
        if np.fabs(new_sigma - old_old_sigma) < 0.001*tol :
            sigma = new_sigma
            break
        old_old_sigma = old_sigma
        old_sigma = new_sigma
        sigma = new_sigma

    return sigma


path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'
X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
X_transposed = unique_rows(X.T)
X = X_transposed.T

n, p = X.shape
X -= X.mean(0)[None, :]
X /= (X.std(0)[None, :] * np.sqrt(n))
print("dims", n,p)

y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
y = y.reshape((y.shape[0],))

sigma = estimate_sigma(X, y, nstep=30, tol=1.e-3)
print("estimated sigma", sigma)
y /= sigma

lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * 1.2
loss = rr.glm.gaussian(X, y)

epsilon = 1. / np.sqrt(n)

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p),
                         weights=dict(zip(np.arange(p), W)), lagrange=1.)

np.random.seed(0)
randomization = randomization.isotropic_gaussian((p,), scale=1.)

M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer='gaussian')

M_est.solve_approx()
active = M_est._overall
active_set = np.asarray([i for i in range(p) if active[i]])
nactive = np.sum(active)
sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
#sys.stderr.write("Observed target" + str(M_est.target_observed)+ "\n")

cov = np.linalg.inv(X[:,active].T.dot(X[:, active])+ 0.01 * epsilon * np.identity(nactive))
target_ridge = cov.dot(X[:, active].T.dot(y))
target_cov_ridge = cov.dot(X[:,active].T.dot(X[:, active])).dot(cov)

ci_naive_ridge = np.zeros((nactive,2))
ci_naive_ridge[:,0] = target_ridge - 1.65* np.sqrt(target_cov_ridge.diagonal())
ci_naive_ridge[:,1] = target_ridge + 1.65* np.sqrt(target_cov_ridge.diagonal())
print("variances", target_cov_ridge.diagonal())
print("unadjusted confidence intervals", sigma* ci_naive_ridge)

