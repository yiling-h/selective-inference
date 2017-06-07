from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

#from selection.frequentist_eQTL.estimator import M_estimator_2step
from selection.randomized.M_estimator import M_estimator
from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage, approximate_conditional_density_2stage

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import instance
from selection.frequentist_eQTL.simes_BH_selection import BH_selection_egenes, simes_selection_egenes
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
#from selection.frequentist_eQTL.instance import instance
from selection.tests.instance import gaussian_instance

class M_estimator_2step(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold,
                 data_simes):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.simes_level = simes_level
        self.index = index
        self.J = J
        self.T_sign = T_sign
        self.threshold = threshold
        self.data_simes = data_simes
        self.nactive_simes = self.threshold.shape[0]

    def solve_approx(self):
        #map from lasso
        #np.random.seed(0)
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
        X_active_inv = np.linalg.inv(X[:,self._overall].T.dot(X[:,self._overall]))
        projection_perp = np.identity(n) - X[:,self._overall].dot(X_active_inv).dot( X[:,self._overall].T)
        score_cov[:nactive, :nactive] = X_active_inv
        score_cov[nactive:, nactive:] = X[:,~self._overall].T.dot(projection_perp).dot(X[:,~self._overall])

        self.score_target_cov = score_cov[:, :nactive]
        self.target_cov = score_cov[:nactive, :nactive]
        self.target_observed = self.observed_score_state[:nactive]
        self.nactive = nactive

        self.B_active_lasso = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive_lasso = self._opt_linear_term[nactive:, :nactive]


        if self.nactive_simes > 1:
            #print(self.nactive_simes, nactive)
            self.score_cov_simes = np.zeros((self.nactive_simes, nactive))
            self.score_cov_simes[0,:] = (X_active_inv.dot(X[:,self._overall].T).dot(X[:,self.index])).T
            self.score_cov_simes[1:,] = (X_active_inv.dot(X[:,self._overall].T).dot(X[:,self.J])).T
            #self.B_active_simes = np.zeros((1,1))
            #self.B_active_simes[0,0] = self.T_sign
            self.B_active_simes = np.identity(1) * self.T_sign
            self.B_inactive_simes = np.zeros((self.nactive_simes-1,1))
            self.inactive_threshold = self.threshold[1:]

        else:
            self.B_active_simes = np.identity(1) * self.T_sign
            self.score_cov_simes = (X_active_inv.dot(X[:, self._overall].T).dot(X[:, self.index]))
            self.inactive_threshold = -1

    def setup_map(self, j):

        self.A_lasso = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic_lasso = self._score_linear_term.dot(self.observed_score_state) - self.A_lasso * self.target_observed[j]

        self.offset_active_lasso = self._opt_affine_term[:self.nactive] + self.null_statistic_lasso[:self.nactive]
        self.offset_inactive_lasso = self.null_statistic_lasso[self.nactive:]

        if self.nactive_simes > 1:
            linear_simes = np.zeros((self.nactive_simes, self.nactive_simes))
            linear_simes[0, 0] = -1.
            linear_simes[1:, 1:] = -np.identity(self.nactive_simes - 1)
            self.A_simes = np.dot(linear_simes, self.score_cov_simes[:, j]) / self.target_cov[j, j]
            self.null_statistic_simes = linear_simes.dot(self.data_simes) - self.A_simes * self.target_observed[j]

            self.offset_active_simes = self.T_sign * self.threshold[0] + self.null_statistic_simes[0]
            self.offset_inactive_simes = self.null_statistic_simes[1:]

        else:
            linear_simes = -1.
            #print("shapes", self.score_cov_simes[j, :].shape, self.target_cov[j, j].shape)
            self.A_simes = linear_simes* (self.score_cov_simes[j] / self.target_cov[j, j])
            self.null_statistic_simes = linear_simes* (self.data_simes) - self.A_simes * self.target_observed[j]
            self.offset_active_simes = self.T_sign * self.threshold[0] + self.null_statistic_simes

def hierarchical_lasso_trial(X,
                             y,
                             beta,
                             sigma,
                             simes_level,
                             index,
                             J,
                             T_sign,
                             threshold,
                             data_simes,
                             bh_level,
                             regime='1',
                             lam_frac=1.,
                             loss='gaussian'):

    from selection.api import randomization

    if regime == '1':
        s=1
    elif regime == '2':
        s=2
    elif regime == '3':
        s=3
    elif regime == '4':
        s=4
    elif regime == '5':
        s=5
    elif regime == '0':
        s=0

    n, p = X.shape
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)

    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    M_est = M_estimator_2step(loss, epsilon, penalty, randomization, simes_level, index, J, T_sign, threshold,
                              data_simes)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed)+ "\n")

    if nactive == 0:
        return None

    else:
        true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

        ci = approximate_conditional_density_2stage(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_covered = np.zeros(nactive, np.bool)
        sel_length = np.zeros(nactive)
        pivots = np.zeros(nactive)

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = np.zeros(nactive)

        for j in xrange(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1
            sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
            print(ci_sel[j, :])
            pivots[j] = ci.approximate_pvalue(j, 0.)

            # naive ci
            if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                naive_covered[j] += 1
            naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]

        #sys.stderr.write("Total adjusted covered" + str(sel_covered.sum()) + "\n")
        #sys.stderr.write("Total naive covered" + str(naive_covered.sum()) + "\n")

        #sys.stderr.write("Pivots" + str(pivots) + "\n")

        power = 0.
        false_discoveries = 0.
        beta_active = beta[active]
        p_BH = BH_q(pivots, bh_level)

        discoveries_active = np.zeros(nactive)
        if p_BH is not None:
            for indx in p_BH[1]:
                discoveries_active[indx] = 1
                if beta_active[indx] != 0.:
                    power += 1.
                else:
                    false_discoveries += 1.

        power = power/float(s)
        fdr = false_discoveries/(max(1.,discoveries_active.sum()))

        sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
        sys.stderr.write("Power" + str(power) + "\n")
        sys.stderr.write("FDR" + str(fdr) + "\n")

        list_results = np.transpose(np.vstack((sel_covered,
                                               sel_length,
                                               pivots,
                                               naive_covered,
                                               naive_pvals,
                                               naive_length,
                                               active_set,
                                               discoveries_active)))


        return list_results


if __name__ == "__main__":

    ###read an input file to set the correct seeds

    BH_genes = np.loadtxt('/home/snigdha/src/selective-inference/selection/frequentist_eQTL/tests/BH_output')
    E_genes = BH_genes[1:]
    E_genes_1 = E_genes[E_genes<600]
    simes_level = BH_genes[0]

    seedn = int(sys.argv[1])
    outdir = sys.argv[2]

    outfile = os.path.join(outdir, "list_result_" + str(seedn) + ".txt")

    ### set parameters
    n = 350
    p = 250
    s = 1
    bh_level = 0.20

    i = int(E_genes_1[seedn])


    np.random.seed(i)
    X, y, beta, nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0, snr=6.)

    simes = simes_selection_egenes(X, y)
    simes_p = simes.simes_p_value()
    sys.stderr.write("simes_p_value" + str(simes_p) + "\n")
    sys.stderr.write("simes level" + str(simes_level) + "\n")


    if simes_p <= simes_level:

        sig_simes = simes.post_BH_selection(simes_level)
        index = sig_simes[2]
        J = sig_simes[1]
        T_sign = sig_simes[3]
        i_0 = sig_simes[0]
        threshold = np.zeros(i_0 + 1)

        ###setting the parameters from Simes
        if i_0 > 0:
            threshold[0] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
            threshold[1:] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (np.arange(i_0) + 1))
            data_simes = np.zeros(i_0 + 1)
            data_simes[0] = X[:, index].T.dot(y)
            data_simes[1:] = X[:, J].T.dot(y)
        else:
            threshold[0] = np.sqrt(2.) * normal.ppf(1. - (simes_level / (2. * p)) * (i_0 + 1))
            data_simes = X[:, index].T.dot(y)

        results = hierarchical_lasso_trial(X,
                                           y,
                                           beta,
                                           sigma,
                                           simes_level,
                                           index,
                                           J,
                                           T_sign,
                                           threshold,
                                           data_simes,
                                           regime='1',
                                           bh_level=0.20,
                                           lam_frac=1.,
                                           loss='gaussian')

        ###save output results
        np.savetxt(outfile, results)







