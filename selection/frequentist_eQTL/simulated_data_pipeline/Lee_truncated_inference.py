from __future__ import print_function
import sys
import os

from selection.algorithms.lasso import lasso
import numpy as np
from scipy.stats import norm as ndist
from scipy.optimize import bisect

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

def lasso_Gaussian(X, y, lam, true_mean):

    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()
    active = soln != 0
    print("Lasso estimator", soln[active])
    nactive = active.sum()
    print("nactive", nactive)

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
    true_val = projection_active.T.dot(true_mean)

    active_set = np.nonzero(active)[0]
    print("active set", active_set)

    active_signs = np.sign(soln[active])
    C = L.constraints
    sel_intervals = np.zeros((nactive, 2))

    coverage_ad = np.zeros(nactive)
    ad_length = np.zeros(nactive)

    if C is not None:
        one_step = L.onestep_estimator
        print("one step", one_step)
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
                sd = np.sqrt((eta * C.covariance.dot(eta)))
                Z = obs / sd
                _pval = 2 * (ndist.sf(min(np.fabs(Z))) - ndist.sf(5)) / (ndist.cdf(5) - ndist.cdf(-5))

                _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                             obs + ndist.ppf(1 - alpha / 2) * sd)

            sel_intervals[i, :] = _interval

            if (sel_intervals[i, 0] <= true_val[i]) and (true_val[i] <= sel_intervals[i, 1]):
                coverage_ad[i] += 1

            ad_length[i] = sel_intervals[i, 1] - sel_intervals[i, 0]

        sel_cov = coverage_ad.sum() / nactive
        ad_len = ad_length.sum() / nactive
        ad_risk = np.power(one_step - true_val, 2.).sum() / nactive

        return sel_cov, ad_len, ad_risk

    else:

        return 0.,0.,0.

if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    ad_cov = 0.
    ad_len = 0.
    ad_risk = 0.
    none = 0.
    for egene in range(len(content)):
        gene = str(content[egene])
        X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
        n, p = X.shape
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))
        X_unpruned = X

        beta = np.load(os.path.join(inpath + "b_" + gene) + ".npy")
        beta = beta.reshape((beta.shape[0],))
        beta = np.sqrt(n) * beta
        true_mean = X_unpruned.dot(beta)
        signal_indices = np.abs(beta) > 0.005

        prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
        prototypes = np.unique(prototypes).astype(int)
        print("prototypes", prototypes.shape[0])
        X = X[:, prototypes]

        y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
        y = y.reshape((y.shape[0],))

        sigma_est = 1.
        y /= sigma_est

        lam_frac = 1.
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

        lasso_results = lasso_Gaussian(X,
                                       y,
                                       lam,
                                       true_mean)

        if lasso_results.sum() !=0:

            ad_cov += lasso_results[0]
            ad_len += lasso_results[1]
            ad_risk += lasso_results[2]

        else:
            none += 1.

        print("\n")
        print("iteration completed", egene)

    final_results = np.zeros(5)
    final_results[0] = ad_cov/float(len(content)-none)
    final_results[1] = ad_len/float(len(content) - none)
    final_results[2] = ad_risk/float(len(content) - none)
    final_results[3] = none
    final_results[4] = signal_indices.sum()

    outfile = os.path.join(outdir + "Lee_et_al_output.txt")
    np.savetxt(outfile, final_results)