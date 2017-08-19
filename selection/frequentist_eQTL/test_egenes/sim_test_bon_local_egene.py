from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues
from scipy.stats.stats import pearsonr

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
                             indices_TS,
                             seed_n = 0,
                             bh_level = 0.10,
                             lam_frac = 1.,
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

    #print("corr of selected X's", np.corrcoef(X[:,active].T))
    #for k in range(nactive):
    #    corr = pearsonr(X[:, index], X[:, active_set[k]])
    #    print("correlation of simes index", corr)

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
        sel_MLE = np.zeros(nactive)
        sel_length = np.zeros(nactive)
        sel_risk = np.zeros(nactive)

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = np.zeros(nactive)
        naive_risk = np.zeros(nactive)


        for j in xrange(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            pivots[j] = ci.approximate_pvalue(j, 0.)

            sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]
            sel_risk[j] = (sel_MLE[j]-true_vec[j])**2.
            sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
            naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]
            naive_risk[j] = (M_est.target_observed[j] - true_vec[j])**2.

            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1
            if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                naive_covered[j] = 1

        select_length = sel_length.sum() / nactive

        naiveci_length = naive_length.sum()/nactive

        p_BH = BH_q(pivots, bh_level)
        power = 0.
        false_discoveries = 0.
        pf = -1* np.ones(nactive)

        discoveries_active = np.zeros(nactive)
        if p_BH is not None:
            for indx in p_BH[1]:
                discoveries_active[indx] = 1
                if indices_TS.shape[0]>1:
                    corr = np.zeros(indices_TS.shape[0])
                    for k in range(indices_TS.shape[0]):
                        corr[k] = pearsonr(X[:, active_set[indx]], X[:, indices_TS[k]])[0]
                    if np.any(corr >= 0.49):
                        power += 1
                    else:
                        false_discoveries += 1.
                else:
                    corr = pearsonr(X[:, active_set[indx]], X[:, indices_TS[0]])[0]
                    if corr >= 0.49:
                        power += 1
                    else:
                        false_discoveries += 1.

        print("lengths", select_length, naiveci_length)
        print("selective p-values", pivots)
        print("selective intervals", ci_sel.T)
        print("naive intervals", ci_naive.T)
        print("risks", sel_risk.sum()/nactive, naive_risk.sum()/nactive)

        pf[0] = power / float(indices_TS.shape[0])
        if discoveries_active.sum()>0.:
            pf[1] = false_discoveries / float(discoveries_active.sum())
        else:
            pf[1] = 0.

        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")
        sys.stderr.write("Total adjusted covered" + str(sel_covered.sum()) + "\n")
        sys.stderr.write("Total naive covered" + str(naive_covered.sum()) + "\n")

        list_results = np.transpose(np.vstack((ci_sel[:, 0],
                                               ci_sel[:, 1],
                                               ci_naive[:, 0],
                                               ci_naive[:, 1],
                                               pivots,
                                               active_set,
                                               sel_covered,
                                               naive_covered,
                                               sel_risk,
                                               naive_risk,
                                               sel_length,
                                               naive_length,
                                               discoveries_active,
                                               pf)))

        sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
        return list_results

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/'

    gene = str("ENSG00000225880.4")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    #X_unpruned = X

    prototypes = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/protoclust_" + gene) + ".txt",
                            delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]
    X_unpruned = X
    print("shape of X", X.shape)

    simulated = np.loadtxt(os.path.join(path + "y_pruned_simulated_" + gene) + ".txt")
    y = simulated[0,:]
    true_mean = simulated[1,:]
    indices_TS = simulated[2,:][simulated[2,:]>-0.5].astype(int)
    print("indices of true signals", indices_TS[0])

    simes_output = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/sim_Test_egenes/Egene_data/simes_" + gene) + ".txt")

    simes_level = (0.10 * 1107)/19555.
    index = int(simes_output[3])
    print("index", index)
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

    try:
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
                                           ratio,
                                           indices_TS,
                                           seed_n=0)


    except ValueError:
        sys.stderr.write("Value error: error try again!" + "\n")
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
                                           ratio,
                                           indices_TS,
                                           seed_n=1)
        try:
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
                                               ratio,
                                               indices_TS,
                                               seed_n=1)
        except ValueError:
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
                                               ratio,
                                               indices_TS,
                                               seed_n=2)

    print(results)