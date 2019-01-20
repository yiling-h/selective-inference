from __future__ import print_function
import sys
import os
from scipy.stats import norm

import numpy as np
import regreg.api as rr

from selection.simulation_prototype.utils import (naive_confidence_intervals,
                                                  BH_q,
                                                  glmnet_sigma)

from selection.simulation_prototype.adjusted_inference import (selection_map2step,
                                                               approximate_conditional_density_2stage)


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
                             sigma_ratio,
                             seed_n=0,
                             lam_frac=1.,
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

    M_est = selection_map2step(loss, epsilon, penalty, randomization, simes_level, index, T_sign,
                               l_threshold, u_threshold, data_simes, sigma_ratio, seed_n)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")

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

            list_results = np.transpose(np.vstack((ci_sel[:, 0],
                                                   ci_sel[:, 1],
                                                   ci_naive[:, 0],
                                                   ci_naive[:, 1],
                                                   active_set,
                                                   sel_covered,
                                                   naive_covered,
                                                   sel_risk,
                                                   naive_risk,
                                                   sel_length,
                                                   naive_length)))

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

            list_results = np.transpose(np.vstack((ci_sel[:, 0],
                                                   ci_sel[:, 1],
                                                   ci_naive[:, 0],
                                                   ci_naive[:, 1],
                                                   active_set,
                                                   sel_covered,
                                                   naive_covered,
                                                   sel_risk,
                                                   naive_risk,
                                                   sel_length,
                                                   naive_length)))

            return list_results


if __name__ == "__main__":

    ###read input files
    inpath = "/Users/snigdhapanigrahi/selective-inference/selection/simulation_prototype/"
    path = "/Users/snigdhapanigrahi/selective-inference/selection/simulation_prototype/data_directory/"
    outdir = "/Users/snigdhapanigrahi/selective-inference/selection/simulation_prototype/inference_output/"
    j = 35

    for egene0 in range(20):
        egene = j+egene0
        content = np.loadtxt(os.path.join(inpath, "eGenes.txt"))
        gene = str(int(content[egene]))
        print("gene", gene)

        X = np.load(os.path.join(path + "X_" + gene) + ".npy")
        n, p = X.shape

        beta = np.load(os.path.join(path + "b_" + gene) + ".npy")
        beta = beta.reshape((beta.shape[0],))
        true_mean = X.dot(beta)
        signal_indices = np.abs(beta) > 0.005

        y = np.load(os.path.join(path + "y_" + gene) + ".npy")
        y = y.reshape((y.shape[0],))

        sigma_est = np.std(y)

        y /= sigma_est
        true_mean /= sigma_est
        simes_output = np.loadtxt(os.path.join(
            "/Users/snigdhapanigrahi/selective-inference/selection/simulation_prototype/bonferroni_output/"
            + "randomized_bon_" + gene) + ".txt")

        simes_level = (0.10 * content.shape[0]) / 500.
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

        data_simes = (sigma_est / sigma_hat) * (X[:, index].T.dot(y))

        sigma = 1.

        ratio = sigma_est / sigma_hat

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
                                           ratio,
                                           seed_n=0)

        outfile = os.path.join(outdir + "inference_" + gene + ".txt")
        np.savetxt(outfile, results)
