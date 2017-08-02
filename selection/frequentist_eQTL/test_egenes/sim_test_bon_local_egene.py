from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.frequentist_eQTL.test_egenes.inference_bon_hierarchical_selection import M_estimator_2step, approximate_conditional_density_2stage

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
                             true_mean,
                             sigma,
                             simes_level,
                             index,
                             T_sign,
                             l_threshold,
                             u_threshold,
                             data_simes,
                             X_unpruned,
                             sigma_ratio,
                             bh_level = 0.10,
                             lam_frac = 1.,
                             loss='gaussian'):

    from selection.api import randomization

    n, p = X.shape
    np.random.seed(0)
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=1.)

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
        true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)
        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

        ci = approximate_conditional_density_2stage(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_covered = np.zeros(nactive, np.bool)
        pivots = np.zeros(nactive)

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_covered = np.zeros(nactive, np.bool)

        for j in xrange(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            pivots[j] = ci.approximate_pvalue(j, 0.)

            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1
            if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                naive_covered[j] += 1

        sel_length = (ci_sel[:, 1] - ci_sel[:, 0]).sum() / nactive

        naive_length = (ci_naive[:,1]- ci_naive[:,0]).sum()/nactive

        p_BH = BH_q(pivots, bh_level)

        discoveries_active = np.zeros(nactive)
        if p_BH is not None:
            for indx in p_BH[1]:
                discoveries_active[indx] = 1

        print("lengths", sel_length, naive_length)
        print("selective intervals", ci_sel.T)
        print("naive intervals", ci_naive.T)
        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")
        sys.stderr.write("Total adjusted covered" + str(sel_covered.sum()) + "\n")
        sys.stderr.write("Total naive covered" + str(naive_covered.sum()) + "\n")

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
    path = '/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/'

    gene = str("ENSG00000162572.15")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X


    prototypes = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/protoclust_" + gene) + ".txt",
                            delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]

    simulated = np.loadtxt(os.path.join(path + "y_simulated_" + gene) + ".txt")
    y = simulated[0,:]
    true_mean = simulated[1,:]
    indices_TS = simulated[2,:][simulated[2,:]>-0.5].astype(int)
    print("indices of true signals", indices_TS)

    simes_output = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/simes_" + gene) + ".txt")

    simes_level = (0.10 * 1472)/19555.
    index = int(simes_output[3])
    T_sign = simes_output[5]

    V = simes_output[0]
    u = simes_output[4]
    sigma_hat = simes_output[6]

    l_threshold = np.sqrt(1+ (0.7**2)) * normal.ppf(1. - min(u, simes_level * (1./ V)) / 2.)
    u_threshold = 10 ** 10

    print("u", u)
    print("l threshold", l_threshold)

    print("ratio", 1./sigma_hat)

    data_simes = (1./sigma_hat)*(X_unpruned[:, index].T.dot(y))

    print("data simes", data_simes)

    sigma = 1.

    ratio = 1./sigma_hat

    results = hierarchical_lasso_trial(X,
                                       y,
                                       true_mean,
                                       sigma,
                                       simes_level,
                                       index,
                                       T_sign,
                                       l_threshold,
                                       u_threshold,
                                       data_simes,
                                       X_unpruned,
                                       ratio)

    print(results)