from __future__ import print_function
import sys
import os

from selection.algorithms.lasso import lasso
import numpy as np
from scipy.stats import norm as ndist
from scipy.optimize import bisect

from rpy2.robjects.packages import importr
from rpy2 import robjects

glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)

                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_minCV = out$lambda.min

                coef = coef(out, s = "lambda.min")
                linear.fit = lm(y~ X[, which(coef>0.001)-1])
                sigma_est = summary(linear.fit)$sigma
                return(sigma_est)
                }''')

    sigma_cv_R = robjects.globalenv['glmnet_cv']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    try:
        sigma_est = sigma_cv_R(r_X, r_y)
    except:
        sigma_est = 1.

    return sigma_est

def restricted_gaussian(Z, interval=[-5., 5.]):
    L_restrict, U_restrict = interval
    Z_restrict = max(min(Z, U_restrict), L_restrict)
    return ((ndist.cdf(Z_restrict) - ndist.cdf(L_restrict)) /
            (ndist.cdf(U_restrict) - ndist.cdf(L_restrict)))


def pivot(L_constraint, Z, U_constraint, S, truth=0):
    F = restricted_gaussian
    if F((U_constraint - truth) / S) != F((L_constraint - truth) / S):
        v = ((F((Z - truth) / S) - F((L_constraint - truth) / S)) /
             (F((U_constraint - truth) / S) - F((L_constraint - truth) / S)))
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
    FU = lambda x: (F(x) - 0.5 * alpha)
    L_conf = bisect(FL, lb, ub)
    U_conf = bisect(FU, lb, ub)
    return np.array([L_conf, U_conf])


def lasso_Gaussian(X, y, lam):
    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()
    active = soln != 0
    print("Lasso estimator", soln[active])
    nactive = active.sum()
    print("nactive", nactive)

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))

    active_set = np.nonzero(active)[0]
    print("active set", active_set)

    active_signs = np.sign(soln[active])
    C = L.constraints
    sel_intervals = np.zeros((nactive, 2))

    unad_length = np.zeros(nactive)
    ad_length = np.zeros(nactive)

    if C is not None:
        one_step = L.onestep_estimator
        print("one step", one_step)
        point_est = projection_active.T.dot(y)
        sd = np.sqrt(np.linalg.inv(X[:, active].T.dot(X[:, active])).diagonal())
        unad_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T

        for k in range(one_step.shape[0]):
            unad_length[k] = unad_intervals[k, 1] - unad_intervals[k, 0]

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
            ad_length[i] = sel_intervals[i, 1] - sel_intervals[i, 0]


        output = np.transpose(np.vstack((sel_intervals[:, 0],
                                         sel_intervals[:, 1],
                                         unad_intervals[:, 0],
                                         unad_intervals[:, 1],
                                         point_est,
                                         ad_length,
                                         unad_length,
                                         nactive * np.ones(nactive))))
        return output

    else:
        return np.transpose(np.vstack((0.,0.,0.,0.,0.,0.,0.,0.)))


if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    # for egene in range(len(content)):
    for egene in range(len(content)):
        gene = str(content[egene])
        if os.path.exists(os.path.join(inpath, "X_" + gene + ".npy")):
            X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
            n, p = X.shape
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))
            X_unpruned = X

            prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
            prototypes = np.unique(prototypes).astype(int)
            print("prototypes", prototypes.shape[0])
            X = X[:, prototypes]

            y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
            y = y.reshape((y.shape[0],))

            sigma_est = glmnet_sigma(X, y)
            print("sigma est", sigma_est)

            y /= sigma_est

            lam_frac = 1.
            lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

            lasso_results = lasso_Gaussian(X,
                                           y,
                                           lam)

            outfile = os.path.join(outdir + "Leeoutput_" + gene + ".txt")
            np.savetxt(outfile, lasso_results)
            sys.stderr.write("Iteration completed" + str(egene) + "\n")
