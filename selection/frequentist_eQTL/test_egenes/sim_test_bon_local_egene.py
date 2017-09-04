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
        try:
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
                sel_risk[j] = (sel_MLE[j] - true_vec[j]) ** 2.
                sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
                naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]
                naive_risk[j] = (M_est.target_observed[j] - true_vec[j]) ** 2.

                if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                    sel_covered[j] = 1
                if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                    naive_covered[j] = 1

            p_BH = BH_q(pivots, bh_level)
            power = 0.
            false_discoveries = 0.
            power_vec = -1 * np.ones(nactive)
            fdr_vec = -1 * np.ones(nactive)

            discoveries_active = np.zeros(nactive)
            true_indices = (signal_indices.nonzero())[0]
            sys.stderr.write("True signal indices" + str(true_indices) + "\n")
            if p_BH is not None:
                for indx in p_BH[1]:
                    discoveries_active[indx] = 1
                    if true_indices.shape[0] > 1:
                        corr = np.zeros(true_indices.shape[0])
                        for k in range(true_indices.shape[0]):
                            corr[k] = pearsonr(X[:, active_set[indx]], X_unpruned[:, true_indices[k]])[0]
                        if np.any(corr >= 0.49):
                            # power += (corr >= 0.49).sum()
                            power += 1.
                        else:
                            false_discoveries += 1.

                    elif true_indices.shape[0] == 1:

                        corr = pearsonr(X[:, active_set[indx]], X_unpruned[:, true_indices[0]])[0]
                        if corr >= 0.49:
                            power += 1
                        else:
                            false_discoveries += 1.
                    else:
                        false_discoveries += 1.

            if true_indices.shape[0] >= 1:
                power_vec[0] = power / float(true_indices.shape[0])
            else:
                power_vec[0] = 0.

            if discoveries_active.sum() > 0.05:
                sys.stderr.write("discoveries" + str(discoveries_active.sum()) + "\n")
                fdr_vec[0] = false_discoveries / float(discoveries_active.sum())
            else:
                fdr_vec[0] = 0.

            sys.stderr.write("True target to be covered" + str(true_vec) + "\n")
            sys.stderr.write("Total adjusted covered" + str(sel_covered.sum()) + "\n")
            sys.stderr.write("Total naive covered" + str(naive_covered.sum()) + "\n")
            sys.stderr.write("Average selective length" + str(sel_length.sum() / nactive) + "\n")
            sys.stderr.write("Average naive length" + str(naive_length.sum() / nactive) + "\n")

            retry = -1 * np.ones(nactive)
            if seed_n == 1:
                retry[0] = 0

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
                                                   power_vec,
                                                   fdr_vec,
                                                   retry)))

            sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
            return list_results

        except ValueError:

            class target_class(object):
                def __init__(self, target_cov):
                    self.target_cov = target_cov
                    self.shape = target_cov.shape

            target = target_class(M_est.target_cov)

            ci_naive = naive_confidence_intervals(target, M_est.target_observed)
            naive_covered = np.zeros(nactive, np.bool)
            naive_length = np.zeros(nactive)
            naive_risk = np.zeros(nactive)

            ci_sel = ci_naive
            for j in xrange(nactive):
                naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]
                naive_risk[j] = (M_est.target_observed[j] - true_vec[j]) ** 2.

                if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                    naive_covered[j] = 1

            sel_covered = naive_covered
            sel_length = naive_length
            sel_risk = naive_risk

            pivots = naive_pvalues(target, M_est.target_observed, np.zeros(nactive))
            p_BH = BH_q(pivots, bh_level)
            power = 0.
            false_discoveries = 0.
            power_vec = -1 * np.ones(nactive)
            fdr_vec = -1 * np.ones(nactive)

            discoveries_active = np.zeros(nactive)
            true_indices = (signal_indices.nonzero())[0]
            sys.stderr.write("True signal indices" + str(true_indices) + "\n")
            if p_BH is not None:
                for indx in p_BH[1]:
                    discoveries_active[indx] = 1
                    if true_indices.shape[0] > 1:
                        corr = np.zeros(true_indices.shape[0])
                        for k in range(true_indices.shape[0]):
                            corr[k] = pearsonr(X[:, active_set[indx]], X_unpruned[:, true_indices[k]])[0]
                        if np.any(corr >= 0.49):
                            # power += (corr >= 0.49).sum()
                            power += 1.
                        else:
                            false_discoveries += 1.

                    elif true_indices.shape[0] == 1:

                        corr = pearsonr(X[:, active_set[indx]], X_unpruned[:, true_indices[0]])[0]
                        if corr >= 0.49:
                            power += 1
                        else:
                            false_discoveries += 1.
                    else:
                        false_discoveries += 1.

            if true_indices.shape[0] >= 1:
                power_vec[0] = power / float(true_indices.shape[0])
            else:
                power_vec[0] = 0.

            if discoveries_active.sum() > 0.05:
                sys.stderr.write("discoveries" + str(discoveries_active.sum()) + "\n")
                fdr_vec[0] = false_discoveries / float(discoveries_active.sum())
            else:
                fdr_vec[0] = 0.

            sys.stderr.write("True target to be covered" + str(true_vec) + "\n")
            sys.stderr.write("Total adjusted covered" + str(sel_covered.sum()) + "\n")
            sys.stderr.write("Total naive covered" + str(naive_covered.sum()) + "\n")
            sys.stderr.write("Average selective length" + str(sel_length.sum() / nactive) + "\n")
            sys.stderr.write("Average naive length" + str(naive_length.sum() / nactive) + "\n")

            retry = -1 * np.ones(nactive)
            if seed_n == 1:
                retry[0] = 0

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
                                                   power_vec,
                                                   fdr_vec,
                                                   retry)))

            sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
            return list_results



if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    egene = int(sys.argv[3])
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    egene = egene
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
    # sigma_est = glmnet_sigma(X, y)
    # print("sigma est", sigma_est)

    y /= sigma_est

    simes_output = np.loadtxt(os.path.join(inpath + "simes_" + gene) + ".txt")

    simes_level = (0.10 * 1770) / 21819.
    index = int(simes_output[3])
    T_sign = simes_output[6]

    V = simes_output[0]
    u = simes_output[5]
    sigma_hat = simes_output[7]

    if u > 10 ** -12.:
        l_threshold = np.sqrt(1 + (0.7 ** 2)) * norm.ppf(1. - min(u, simes_level * (1. / V)) / 2.)
    else:
        l_threshold = np.sqrt(1 + (0.7 ** 2)) * norm.ppf(1. - (simes_level * (1. / V) / 2.))

    u_threshold = 10 ** 7.

    data_simes = (sigma_est / sigma_hat) * (X_unpruned[:, index].T.dot(y))

    sigma = 1.

    ratio = sigma_est / sigma_hat

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
                                           signal_indices,
                                           seed_n=0)

        outfile = os.path.join(outdir + "inference_" + gene + ".txt")
        np.savetxt(outfile, results)

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
                                           signal_indices,
                                           seed_n=1)

        outfile = os.path.join(outdir + "inference_" + gene + ".txt")
        np.savetxt(outfile, results)