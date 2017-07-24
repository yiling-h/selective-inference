import os, numpy as np, pandas, statsmodels.api as sm
import time
import sys

import regreg.api as rr
from selection.algorithms.lasso import lasso
#from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
#from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
#    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.ridge_target import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.estimator import M_estimator_exact
from scipy.stats import norm as ndist
from scipy.optimize import bisect

def selection(X, y, random_Z, randomization_scale=1., sigma=None, method="theoretical"):
    n, p = X.shape
    loss = rr.glm.gaussian(X,y)
    epsilon = 1. / np.sqrt(n)
    lam_frac = 0.8
    if sigma is None:
        sigma = 1.
    if method == "theoretical":
        lam = 1. * sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p), weights = dict(zip(np.arange(p), W)), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -randomization_scale * random_Z, 0)


    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return None
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE = initial_soln[active]
    subgradient = -(initial_grad+epsilon*initial_soln-randomization_scale*random_Z)
    cube = subgradient[~active]/lam
    return lam, epsilon, active, betaE, cube, initial_soln

def restricted_gaussian(Z, interval=[-5.,5.]):
    L_restrict, U_restrict = interval
    Z_restrict = max(min(Z, U_restrict), L_restrict)
    return ((ndist.cdf(Z_restrict) - ndist.cdf(L_restrict)) /
            (ndist.cdf(U_restrict) - ndist.cdf(L_restrict)))

def pivot(L_constraint, Z, U_constraint, S, truth=0):
    F = restricted_gaussian
    if F((U_constraint - truth) / S) != F((L_constraint -  truth) / S):
        v = ((F((Z-truth)/S) - F((L_constraint-truth)/S)) /
             (F((U_constraint-truth)/S) - F((L_constraint-truth)/S)))
    elif F((U_constraint - truth) / S) < 0.1:
        v = 1
    else:
        v = 0
    return v

def equal_tailed_interval(L_constraint, Z, U_constraint, S, alpha=0.05):

    lb = Z - 5 * S
    ub = Z + 5 * S

    def F(param):
        return pivot(L_constraint, Z, U_constraint, S, truth=param)

    FL = lambda x: (F(x) - (1 - 0.5 * alpha))
    FU = lambda x: (F(x) - 0.5* alpha)
    L_conf = bisect(FL, lb, ub)
    U_conf = bisect(FU, lb, ub)
    return np.array([L_conf, U_conf])


if __name__ == "__main__":

    #np.random.seed(2)
    path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

    X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")

    prototypes = np.loadtxt("/Users/snigdhapanigrahi/Results_bayesian/Egene_data/prototypes.txt", delimiter='\t')
    prototypes = prototypes.astype(int)-1
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]
    n, p = X.shape
    print("dimensions", n, p)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))

    y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
    y = y.reshape((y.shape[0],))

    sigma = 0.40355257294593277
    print("estimated sigma", sigma)
    y /= sigma

    lam_frac = 0.8
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))
    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()

    active = soln != 0
    nactive = active.sum()
    print("nactive", nactive)

    active_set = np.nonzero(active)[0]
    print("active set", active_set)

    active_signs = np.sign(soln[active])
    C = L.constraints
    sel_intervals = np.zeros((nactive,2))

    if C is not None:
        one_step = L.onestep_estimator
        for i in range(one_step.shape[0]):
            eta = np.zeros_like(one_step)
            eta[i] = active_signs[i]
            alpha = 0.1

            if C.linear_part.shape[0] > 0:  # there were some constraints
                L, Z, U, S = C.bounds(eta, one_step)
                _pval = pivot(L, Z, U, S)
                # two-sided
                _pval = 2 * min(_pval, 1 - _pval)

                L, Z, U, S = C.bounds(eta, one_step)
                _interval = equal_tailed_interval(L, Z, U, S, alpha=alpha)
                _interval = sorted([_interval[0] * active_signs[i],
                                    _interval[1] * active_signs[i]])

            else:
                obs = (eta * one_step).sum()
                ## jelena: should be this sd = np.sqrt(np.dot(eta.T, C.covariance.dot(eta))), no?
                sd = np.sqrt((eta * C.covariance.dot(eta)))
                Z = obs / sd
                _pval = 2 * (ndist.sf(min(np.fabs(Z))) - ndist.sf(5)) / (ndist.cdf(5) - ndist.cdf(-5))

                _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                             obs + ndist.ppf(1 - alpha / 2) * sd)

            sel_intervals[i,:] = _interval

    sel_intervals = sigma* sel_intervals
    print("intervals", sel_intervals)
    print("length", (sel_intervals[:,1]-sel_intervals[:,0]).sum()/nactive)








