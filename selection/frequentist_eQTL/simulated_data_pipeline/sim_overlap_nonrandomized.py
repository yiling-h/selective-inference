from __future__ import print_function
import sys
import os
from scipy.stats import norm

import numpy as np
import regreg.api as rr

from selection.randomized.M_estimator import M_estimator
from scipy.stats.stats import pearsonr

from rpy2.robjects.packages import importr
from rpy2 import robjects

glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()


class M_estimator_2step(M_estimator):
    def __init__(self, loss, epsilon, penalty, randomization, simes_level, index, T_sign, l_threshold, u_threshold,
                 data_simes, X_unpruned, sigma_ratio):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.simes_level = simes_level
        self.index = index
        self.T_sign = T_sign
        self.data_simes = data_simes
        self.l_threshold = l_threshold
        self.u_threshold = u_threshold
        self.X_unpruned = X_unpruned
        self.sigma_ratio = sigma_ratio
        self.simes_randomization = 0.7

    def solve_approx(self):
        self.solve()
        (_opt_linear_term, _opt_affine_term) = self.opt_transform
        self._opt_linear_term = np.concatenate(
            (_opt_linear_term[self._overall, :], _opt_linear_term[~self._overall, :]), 0)
        self._opt_affine_term = np.concatenate((_opt_affine_term[self._overall], _opt_affine_term[~self._overall]), 0)
        self.opt_transform = (self._opt_linear_term, self._opt_affine_term)

        (_score_linear_term, _) = self.score_transform
        self._score_linear_term = np.concatenate(
            (_score_linear_term[self._overall, :], _score_linear_term[~self._overall, :]), 0)
        self.score_transform = (self._score_linear_term, np.zeros(self._score_linear_term.shape[0]))
        self.feasible_point_lasso = np.abs(self.initial_soln[self._overall])

        lagrange = []
        for key, value in self.penalty.weights.iteritems():
            lagrange.append(value)
        lagrange = np.asarray(lagrange)
        self.inactive_lagrange = lagrange[~self._overall]

        X, _ = self.loss.data
        n, p = X.shape
        self.p = p

        nactive = self._overall.sum()

        score_cov = np.zeros((p, p))
        X_active_inv = np.linalg.inv(X[:, self._overall].T.dot(X[:, self._overall]))
        projection_perp = np.identity(n) - X[:, self._overall].dot(X_active_inv).dot(X[:, self._overall].T)
        score_cov[:nactive, :nactive] = X_active_inv
        score_cov[nactive:, nactive:] = X[:, ~self._overall].T.dot(projection_perp).dot(X[:, ~self._overall])

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active_lasso = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive_lasso = self._opt_linear_term[nactive:, :nactive]

        self.score_cov_simes = self.sigma_ratio * (
            X_active_inv.dot(X[:, self._overall].T).dot(self.X_unpruned[:, self.index]))

    def setup_map(self, j):
        self.A_lasso = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic_lasso = self._score_linear_term.dot(self.observed_score_state) - self.A_lasso * \
                                                                                             self.target_observed[j]

        self.offset_active_lasso = self._opt_affine_term[:self.nactive] + self.null_statistic_lasso[:self.nactive]
        self.offset_inactive_lasso = self.null_statistic_lasso[self.nactive:]

        linear_simes = -self.T_sign
        self.A_simes = linear_simes * (self.score_cov_simes[j] / self.target_cov[j, j])
        self.null_statistic_simes = linear_simes * (self.data_simes) - self.A_simes * self.target_observed[j]

        self.offset_simes = self.null_statistic_simes


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
                             signal_indices,
                             seed_n=0,
                             bh_level=0.10,
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

    M_est = M_estimator_2step(loss, epsilon, penalty, randomization, simes_level, index, T_sign,
                              l_threshold, u_threshold, data_simes, X_unpruned, sigma_ratio)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")

    no_randomization = randomization.isotropic_gaussian((p,), scale=.001)
    M_est_nonrand = M_estimator_2step(loss, epsilon, penalty, no_randomization, simes_level, index, T_sign,
                                      l_threshold, u_threshold, data_simes, X_unpruned, sigma_ratio)
    M_est_nonrand.solve_approx()
    active_nonrand = M_est_nonrand._overall
    active_set_nonrand = np.asarray([i for i in range(p) if active_nonrand[i]])
    nactive_nonrand = np.sum(active_nonrand)
    sys.stderr.write("number of active selected by non-randomized lasso" + str(nactive_nonrand) + "\n")
    sys.stderr.write("Active set selected by non-randomized lasso" + str(active_set_nonrand) + "\n")

    true_indices = (signal_indices.nonzero())[0]
    corr_rand = np.zeros(true_indices.shape[0])
    corr_nonrand = np.zeros(true_indices.shape[0])

    true_sel = []
    false_sel = []
    if nactive>=1:
        for i in range(nactive):
            if true_indices.shape[0] >= 1:
                for k in range(true_indices.shape[0]):
                    corr_rand[k] = pearsonr(X[:, active_set[i]], X_unpruned[:, true_indices[k]])[0]
                if np.any(corr_rand >= 0.49):
                    true_sel.append(active_set[i])
                else:
                    false_sel.append(active_set[i])
            else:
                false_sel.append(active_set[i])

    elif nactive == 1:
        if true_indices.shape[0] >= 1:
            for k in range(true_indices.shape[0]):
                corr_rand[k] = pearsonr(X[:, active_set[0]], X_unpruned[:, true_indices[k]])[0]
            if np.any(corr_rand >= 0.49):
                true_sel.append(active_set[0])
            else:
                false_sel.append(active_set[0])
        else:
            false_sel.append(active_set[0])

    true_sel = np.asarray(true_sel)
    power_rand = true_sel.shape[0]/max(1., float(true_indices.shape[0]))
    false_sel = np.asarray(false_sel)

    true_nonrand_sel = []
    false_nonrand_sel = []

    if nactive_nonrand>1:
        for j in range(nactive_nonrand):
            if true_indices.shape[0] >= 1:
                for l in range(true_indices.shape[0]):
                    corr_nonrand[l] = pearsonr(X[:, active_set_nonrand[j]], X_unpruned[:, true_indices[l]])[0]
                if np.any(corr_nonrand >= 0.49):
                    true_nonrand_sel.append(active_set_nonrand[j])
                else:
                    false_nonrand_sel.append(active_set_nonrand[j])
            else:
                false_sel.append(active_set_nonrand[j])
    elif nactive_nonrand == 1:
        if true_indices.shape[0] >= 1:
            for l in range(true_indices.shape[0]):
                corr_nonrand[l] = pearsonr(X[:, active_set_nonrand[0]], X_unpruned[:, true_indices[l]])[0]
            if np.any(corr_nonrand >= 0.49):
                true_nonrand_sel.append(active_set_nonrand[0])
            else:
                false_nonrand_sel.append(active_set_nonrand[0])
        else:
            false_sel.append(active_set_nonrand[0])

    true_nonrand_sel = np.asarray(true_nonrand_sel)
    power_nonrand = true_nonrand_sel.shape[0] / max(1., float(true_indices.shape[0]))
    false_nonrand_sel = np.asarray(false_nonrand_sel)

    diff_rand = np.setdiff1d(active_set, active_set_nonrand)
    diff_non_rand = np.setdiff1d(active_set_nonrand, active_set)

    diff_true_sel_rand = np.setdiff1d(true_sel, true_nonrand_sel)
    diff_true_sel_nonrand = np.setdiff1d(true_nonrand_sel, true_sel)

    diff_false_sel_rand = np.setdiff1d(false_sel, false_nonrand_sel)
    diff_false_sel_nonrand = np.setdiff1d(false_nonrand_sel, false_sel)

    list_results = np.transpose(np.vstack((diff_rand.shape[0],
                                           diff_non_rand.shape[0],
                                           diff_true_sel_rand.shape[0],
                                           diff_true_sel_nonrand.shape[0],
                                           diff_false_sel_rand.shape[0],
                                           diff_false_sel_nonrand.shape[0],
                                           true_sel.shape[0],
                                           true_nonrand_sel.shape[0],
                                           power_rand,
                                           power_nonrand,
                                           nactive,
                                           nactive_nonrand,
                                           )))

    return list_results


if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    egene = int(sys.argv[3])
    egene = int(egene * 100)
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    for k in range(100):

        sys.stderr.write("iteration started" + str(k) + "\n")
        gene = str(content[(egene + k)])

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

        sigma_ratio = sigma_est / sigma_hat

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
                                           sigma_ratio,
                                           signal_indices)

        outfile = os.path.join(outdir + "nonrand_sel_" + gene + ".txt")
        np.savetxt(outfile, results)

