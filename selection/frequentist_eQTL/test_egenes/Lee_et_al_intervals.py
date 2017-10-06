import os, numpy as np, pandas, statsmodels.api as sm
import time
import sys

import regreg.api as rr
from selection.algorithms.lasso import lasso
from selection.randomized.api import randomization
from selection.reduced_optimization.estimator import M_estimator_exact
from scipy.stats import norm as ndist
from scipy.optimize import bisect

def BH_q(p_value, level):

    m = p_value.shape[0]
    p_sorted = np.sort(p_value)
    indices = np.arange(m)
    indices_order = np.argsort(p_value)

    if np.any(p_sorted - np.true_divide(level*(np.arange(m)+1.),m)<=np.zeros(m)):
        order_sig = np.max(indices[p_sorted- np.true_divide(level*(np.arange(m)+1.),m)<=0])
        sig_pvalues = indices_order[:(order_sig+1)]
        return p_sorted[:(order_sig+1)], sig_pvalues

    else:
        return None


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
    path = '/Users/snigdhapanigrahi/Test_bon_egenes/Egene_data/'
    gene = str("ENSG00000215915.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X

    prototypes = np.loadtxt(
        os.path.join("/Users/snigdhapanigrahi/Test_bon_egenes/Egene_data/protoclust_" + gene) + ".txt",
        delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]
    n, p = X.shape

    y = np.load(os.path.join(path + "y_" + gene) + ".npy")
    y = y.reshape((y.shape[0],))
    # sigma_est =  0.3234533
    # sigma_est = 0.5526097
    sigma_est = 0.4303074
    y /= sigma_est

    np.random.seed(0)
    lam_frac = .8
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
    sel_intervals = np.zeros((nactive, 2))
    sel_pvalues = np.zeros(nactive)

    OLS = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(y)
    var = np.linalg.inv(X[:, active].T.dot(X[:, active]))
    naive_ci = np.vstack([OLS - 1.65 * (np.sqrt(var.diagonal())),OLS + 1.65 * (np.sqrt(var.diagonal()))])

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

            sel_intervals[i, :] = _interval
            sel_pvalues[i] = _pval

    sel_intervals = sigma_est * sel_intervals
    p_BH = BH_q(sel_pvalues, 0.10)
    discoveries_active = np.zeros(nactive)
    if p_BH is not None:
        for indx in p_BH[1]:
            discoveries_active[indx] = 1

    discoveries = np.nonzero(discoveries_active)[0]
    naive_intervals = sigma_est * naive_ci

    print("selective intervals", sel_intervals.T)
    print("naive intervals", naive_intervals)
    print("selective and unadjusted lengths", (sel_intervals[:, 1] - sel_intervals[:, 0]).sum() / nactive,
          (naive_intervals[1,:] - naive_intervals[0,:]).sum() / nactive)
    print("discoveries", discoveries_active)
