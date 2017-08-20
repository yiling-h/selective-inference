from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.frequentist_eQTL.test_egenes.inference_bon_hierarchical_selection import M_estimator_2step, approximate_conditional_density_2stage

from rpy2.robjects.packages import importr
from rpy2 import robjects
glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def naive_pvalues(target, observed, parameter):
    pvalues = np.zeros(target.shape[0])
    for j in range(target.shape[0]):
        sigma = np.sqrt(target.target_cov[j, j])
        pval = normal.cdf((observed[j]-parameter[j])/sigma)
        pvalues[j] = 2*min(pval, 1-pval)
    return pvalues

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

    sigma_est = sigma_cv_R(r_X, r_y)
    return sigma_est

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


def hierarchical_lasso_trial(X,
                             y,
                             sigma,
                             simes_level,
                             index,
                             T_sign,
                             l_threshold,
                             u_threshold,
                             data_simes,
                             X_unpruned,
                             sigma_ratio,
                             seed_n = 0,
                             bh_level = 0.10,
                             lam_frac = 1.2,
                             loss='gaussian'):

    from selection.api import randomization

    n, p = X.shape
    np.random.seed(seed_n)
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=.7)

    M_est = M_estimator_2step(loss, epsilon, penalty, randomization, simes_level, index, T_sign,
                              l_threshold, u_threshold, data_simes, X_unpruned, sigma_ratio)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed)+ "\n")

    if nactive == 0:
        return None

    else:

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_length = (ci_naive[:, 1] - ci_naive[:, 0]).sum() / nactive

        try:
            ci = approximate_conditional_density_2stage(M_est)
            ci.solve_approx()

            ci_sel = np.zeros((nactive, 2))
            pivots = np.zeros(nactive)
            sel_MLE = np.zeros(nactive)

            for j in xrange(nactive):
                ci_sel[j, :] = np.array(ci.approximate_ci(j))
                pivots[j] = ci.approximate_pvalue(j, 0.)
                sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]

            sel_length = (ci_sel[:, 1] - ci_sel[:, 0]).sum() / nactive

        except ValueError:
            ci_sel = ci_naive
            pivots = naive_pvalues(target, M_est.target_observed, np.zeros(nactive))
            sel_MLE = M_est.target_observed

            sel_length = (ci_sel[:, 1] - ci_sel[:, 0]).sum() / nactive

        p_BH = BH_q(pivots, bh_level)

        discoveries_active = np.zeros(nactive)
        if p_BH is not None:
            for indx in p_BH[1]:
                discoveries_active[indx] = 1

        print("lengths", sel_length, naive_length)
        print("selective intervals", ci_sel.T)
        print("selective MLE", sel_MLE)
        print("uandjusted MLE", M_est.target_observed)
        print("naive intervals", ci_naive.T)

        list_results = np.transpose(np.vstack((ci_sel[:, 0],
                                               ci_sel[:, 1],
                                               ci_naive[:, 0],
                                               ci_naive[:, 1],
                                               pivots,
                                               active_set,
                                               discoveries_active)))

        sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
        return list_results


if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/Egene_data/'

    gene = str("ENSG00000225630.1")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X


    prototypes = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/Test_egenes/Egene_data/protoclust_" + gene) + ".txt",
                            delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]

    y = np.load(os.path.join(path + "y_" + gene) + ".npy")
    y = y.reshape((y.shape[0],))

    sigma_est = glmnet_sigma(X, y)
    print("sigma est", sigma_est)

    y /= sigma_est

    simes_output = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/Test_egenes/Egene_data/simes_" + gene) + ".txt")

    simes_level = (0.10 * 2167)/21819.
    index = int(simes_output[2])
    T_sign = simes_output[5]

    V = simes_output[0]
    u = simes_output[4]
    sigma_hat = simes_output[6]

    if u > 10 ** -12.:
        l_threshold = np.sqrt(1+ (0.7**2)) * normal.ppf(1. - min(u, simes_level * (1./ V)) / 2.)
    else:
        l_threshold = np.sqrt(1 + (0.7 ** 2)) * normal.ppf(1. -(simes_level * (1./ V)/2.))

    u_threshold = 10 ** 10

    print("u", u)
    print("l threshold", l_threshold)

    print("ratio", sigma_est/sigma_hat)

    data_simes = (sigma_est/sigma_hat)*(X_unpruned[:, index].T.dot(y))

    print("data simes", data_simes)

    sigma = 1.

    ratio = sigma_est/sigma_hat

    results = hierarchical_lasso_trial(X,
                                       y,
                                       sigma,
                                       simes_level,
                                       index,
                                       T_sign,
                                       l_threshold,
                                       u_threshold,
                                       data_simes,
                                       X_unpruned,
                                       ratio,
                                       seed_n=0)

    print(results)