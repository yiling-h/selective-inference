import os
from scipy.stats import norm as normal
import sys

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

def correlation(X, subset_1, subset_2):

    new_predictors = np.setdiff1d(subset_1, subset_2)
    subset = []
    corr = np.zeros((new_predictors.shape[0], subset_2.shape[0]))
    for i in range(new_predictors.shape[0]):
        for j in range(subset_2.shape[0]):
            corr[i,j] = np.correlate(X[:,new_predictors[i]], X[:,subset_2[j]])

        print("corr", corr[i, :])
        if np.all(np.abs(corr[i,:])<0.5):
            subset.append(new_predictors[i])

    print("new predictors", new_predictors)
    return subset

def randomized_lasso_egene_trial(X,
                                 y,
                                 sigma,
                                 seedn,
                                 lam_frac_1 = 1.,
                                 lam_frac_2 = 1.2,
                                 loss='gaussian'):

    from selection.api import randomization

    np.random.seed(seedn)

    n, p = X.shape
    if loss == "gaussian":
        lam_1 = lam_frac_1 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        lam_2 = lam_frac_2 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W_1 = np.ones(p) * lam_1
    penalty_1 = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W_1)), lagrange=1.)

    W_2 = np.ones(p) * lam_2
    penalty_2 = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W_2)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    M_est_1 = M_estimator_exact(loss, epsilon, penalty_1, randomization)
    M_est_1.solve_approx()
    active_1 = M_est_1._overall
    active_set_1 = np.asarray([i for i in range(p) if active_1[i]])
    nactive_1 = np.sum(active_1)

    print("lasso selects at smaller penalty", nactive_1)

    sys.stderr.write("number of active selected by lasso at smaller penalty" + str(nactive_1) + "\n")
    sys.stderr.write("Active set selected by lasso at smaller penalty" + str(active_set_1) + "\n")

    M_est_2 = M_estimator_exact(loss, epsilon, penalty_2, randomization)
    M_est_2.solve_approx()
    active_2 = M_est_2._overall
    active_set_2 = np.asarray([i for i in range(p) if active_2[i]])
    nactive_2 = np.sum(active_2)

    print("lasso selects at larger penalty", nactive_2)

    sys.stderr.write("number of active selected by lasso at larger penalty" + str(nactive_2) + "\n")
    sys.stderr.write("Active set selected by lasso at larger penalty" + str(active_set_2) + "\n")

    ind_subset = correlation(X, active_set_1, active_set_2)

    return ind_subset


if __name__ == "__main__":

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

    seedn = 0

    random_lasso = randomized_lasso_egene_trial(X,
                                                y,
                                                1.,
                                                seedn)

    print("ind subset", random_lasso)