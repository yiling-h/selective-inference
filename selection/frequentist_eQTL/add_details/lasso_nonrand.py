import os, numpy as np
from selection.algorithms.lasso import lasso
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

    coverage_unad = np.zeros(nactive)

    if C is not None:
        one_step = L.onestep_estimator
        print("one step", one_step)

        point_est = projection_active.T.dot(y)
        sd = np.sqrt(np.linalg.inv(X[:, active].T.dot(X[:, active])).diagonal())
        unad_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T
        unad_length = (unad_intervals[:,1]- unad_intervals[:,0]).sum() / nactive
        unad_risk = np.power(point_est- true_val, 2.).sum() / nactive

        for i in range(one_step.shape[0]):
            if (unad_intervals[i, 0] <= true_val[i]) and (true_val[i] <= unad_intervals[i, 1]):
                coverage_unad[i] = 1
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

        sel_cov = coverage_ad.sum() /float(nactive)
        ad_len = ad_length.sum() /float(nactive)
        ad_risk = np.power(one_step - true_val, 2.).sum() /float(nactive)

        return sel_cov, ad_len, ad_risk, coverage_unad.sum() /float(nactive), unad_length, unad_risk

    else:

        return 0.,0.,0., 0.,0.,0.

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/sim_Egene_data/'

    gene = str("ENSG00000187642.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X

    prototypes = np.loadtxt(os.path.join(path + "protoclust_" + gene) + ".txt", delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])
    X = X[:, prototypes]
    n, p = X.shape
    print("shape of X", n, p)

    sel_risk = 0.
    sel_covered = 0.
    sel_length = 0.

    unad_covered = 0.
    unad_length = 0.

    count = 0
    lam_frac = 0.8
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

    for seed_n in range(100):

        np.random.seed(seed_n)
        y = np.random.standard_normal(n)
        true_mean = np.zeros(n)
        results = lasso_Gaussian(X, y, lam, true_mean)
        if results[0] == 0 and results[1]==0 and results[2]==0:
            count += 1
        else:
            print("results", results)
            sel_covered += results[0]
            sel_risk += results[2]
            sel_length += results[1]
            unad_covered += results[3]
            unad_length += results[4]
            print("iteration completed", seed_n)

    print("results", count, sel_covered, sel_risk, sel_length, unad_covered, unad_length)
