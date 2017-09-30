from __future__ import print_function
import sys
import os
from scipy.stats import norm

import numpy as np
import regreg.api as rr

from selection.randomized.M_estimator import M_estimator

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
                             seed_n=0,
                             lam_frac=1.2,
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
    sys.stderr.write("Observed target" + str(M_est.target_observed) + "\n")

    no_randomization = randomization.isotropic_gaussian((p,), scale=.001)

    M_est_nonrand = M_estimator_2step(loss, epsilon, penalty, no_randomization, simes_level, index, T_sign,
                                      l_threshold, u_threshold, data_simes, X_unpruned, sigma_ratio)
    M_est_nonrand.solve_approx()
    active_nonrand = M_est_nonrand._overall
    active_set_nonrand = np.asarray([i for i in range(p) if active_nonrand[i]])
    nactive_nonrand = np.sum(active_nonrand)
    sys.stderr.write("number of active selected by non-randomized lasso" + str(nactive_nonrand) + "\n")
    sys.stderr.write("Active set selected by non-randomized lasso" + str(active_set_nonrand) + "\n")
    sys.stderr.write("Observed target by non-randomized" + str(M_est_nonrand.target_observed) + "\n")
    diff_rand = np.setdiff1d(active_set, active_set_nonrand)
    diff_non_rand = np.setdiff1d(active_set_nonrand, active_set)


    list_results = np.transpose(np.vstack((diff_rand.shape[0],
                                           diff_non_rand.shape[0],
                                           diff_non_rand.shape[0]/np.max(float(nactive_nonrand),1.),
                                           nactive,
                                           nactive_nonrand)))

    return list_results

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/Egene_data/'

    gene = str("ENSG00000228697.1")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    print("dimensions", p)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X


    prototypes = np.loadtxt(os.path.join("/Users/snigdhapanigrahi/Test_egenes/Egene_data/protoclust_" + gene) + ".txt",
                            delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)

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
        l_threshold = np.sqrt(1+ (0.7**2)) * norm.ppf(1. - min(u, simes_level * (1./ V)) / 2.)
    else:
        l_threshold = np.sqrt(1 + (0.7 ** 2)) * norm.ppf(1. -(simes_level * (1./ V)/2.))

    u_threshold = 10 ** 7.

    print("u", u)
    print("l threshold", l_threshold)

    print("ratio", sigma_est/sigma_hat)

    data_simes = (sigma_est/sigma_hat)*(X_unpruned[:, index].T.dot(y))

    print("data simes", data_simes)

    sigma = 1.

    sigma_ratio = sigma_est/sigma_hat

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
                                       sigma_ratio,
                                       seed_n=0)

    print("results", results)

