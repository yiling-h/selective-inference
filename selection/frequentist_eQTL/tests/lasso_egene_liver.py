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

from selection.frequentist_eQTL.approx_confidence_intervals_scaled import neg_log_cube_probability,\
    approximate_conditional_prob, \
    approximate_conditional_density
#from selection.frequentist_eQTL.approx_ci_randomized_lasso import approximate_conditional_density

def BH_q(p_value, level):

    m = p_value.shape[0]
    p_sorted = np.sort(p_value)
    indices = np.arange(m)
    indices_order = np.argsort(p_value)

    #print("sorted p values", p_sorted-np.true_divide(level*(np.arange(m)+1.),2.*m))
    if np.any(p_sorted - np.true_divide(level*(np.arange(m)+1.),m)<=np.zeros(m)):
        order_sig = np.max(indices[p_sorted- np.true_divide(level*(np.arange(m)+1.),m)<=0])
        sig_pvalues = indices_order[:(order_sig+1)]
        return p_sorted[:(order_sig+1)], sig_pvalues

    else:
        return None

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

def randomized_lasso_egene_trial(X,
                                 y,
                                 sigma,
                                 bh_level,
                                 seedn,
                                 lam_frac = 1.,
                                 loss='gaussian'):

    from selection.api import randomization

    np.random.seed(seedn)

    n, p = X.shape
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)


    randomization = randomization.isotropic_gaussian((p,), scale=0.7)

    M_est = M_estimator_exact(loss, epsilon, penalty, randomization)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    print("lasso selects", nactive)

    if nactive == 0:
        return None
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")

    class target_class(object):
        def __init__(self, target_cov):
            self.target_cov = target_cov
            self.shape = target_cov.shape

    target = target_class(M_est.target_cov)

    ci_naive = naive_confidence_intervals(target, M_est.target_observed)
    naive_length = np.zeros(nactive)

    ci = approximate_conditional_density(M_est)
    ci.solve_approx()

    ci_sel = np.zeros((nactive, 2))
    sel_length = np.zeros(nactive)
    pivots = np.zeros(nactive)

    for j in xrange(nactive):
        ci_sel[j, :] = np.array(ci.approximate_ci(j))
        sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
        naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]
        pivots[j] = ci.approximate_pvalue(j, 0.)

    print("selective intervals", ci_sel[:,0], ci_sel[:,1])

    mle = np.zeros(nactive)
    for j in range(nactive):
        mle[j] = ci.approx_MLE_solver(j)[0]

    print("selective MLE", mle)

    p_BH = BH_q(pivots, bh_level)
    discoveries_active = np.zeros(nactive)
    if p_BH is not None:
        for indx in p_BH[1]:
            discoveries_active[indx] = 1

    list_results = np.transpose(np.vstack((ci_sel[:,0],
                                           ci_sel[:, 1],
                                           ci_naive[:,0],
                                           ci_naive[:,1],
                                           sel_length,
                                           naive_length,
                                           discoveries_active)))

    print("list of results", list_results)
    return list_results


if __name__ == "__main__":

    path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

    X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
    n, p = X.shape

    prototypes = np.loadtxt("/Users/snigdhapanigrahi/Results_bayesian/Egene_data/prototypes.txt", delimiter='\t')
    prototypes = prototypes.astype(int) - 1
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))

    n, p = X.shape
    print("dims", n, p)

    y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
    y = y.reshape((y.shape[0],))

    sigma = 0.40355257294593277
    #sigma = estimate_sigma(X, y, nstep=20, tol=1.e-5)
    #print("estimated sigma", sigma)
    y /= sigma

    bh_level = 0.10
    seedn = 0

    random_lasso = randomized_lasso_egene_trial(X,
                                                y,
                                                1.,
                                                bh_level,
                                                seedn)


# cov = np.linalg.inv(X[:,active].T.dot(X[:, active])+ 0. * epsilon * np.identity(nactive))
# target_ridge = cov.dot(X[:, active].T.dot(y))
# target_cov_ridge = cov.dot(X[:,active].T.dot(X[:, active])).dot(cov)
#
# ci_naive_ridge = np.zeros((nactive,2))
# ci_naive_ridge[:,0] = target_ridge - 1.65* np.sqrt(target_cov_ridge.diagonal())
# ci_naive_ridge[:,1] = target_ridge + 1.65* np.sqrt(target_cov_ridge.diagonal())
# print("variances", target_cov_ridge.diagonal())
# print("unadjusted confidence intervals", sigma* ci_naive_ridge)

