from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

#from selection.frequentist_eQTL.estimator import M_estimator_2step
from selection.randomized.M_estimator import M_estimator
#from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage, approximate_conditional_density_2stage

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

class approximate_conditional_prob_2stage(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 map,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.map = map
        self.q_lasso = map.p - map.nactive
        self.inactive_conjugate = self.active_conjugate = map.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = self.map.inactive_lagrange

        if self.map.nactive_simes > 1:
            self.inactive_threshold = self.map.inactive_threshold
            self.q_simes = self.map.nactive_simes-1

        feasible_point = np.ones(self.map.nactive + 1)
        feasible_point[1:] = self.map.feasible_point_lasso
        self.feasible_point = feasible_point

        rr.smooth_atom.__init__(self,
                                (self.map.nactive+1,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.feasible_point,
                                coef=coef)

        self.coefs[:] = feasible_point

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.map.nactive+1)


    def sel_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        arg_simes = np.zeros(self.map.nactive + 1, bool)
        arg_simes[0] = 1
        arg_lasso = ~arg_simes

        data_lasso = np.squeeze(self.t *  self.map.A_lasso)

        offset_active_lasso = self.map.offset_active_lasso + data_lasso[:self.map.nactive]
        offset_inactive_lasso = self.map.offset_inactive_lasso + data_lasso[self.map.nactive:]

        _active_lasso = rr.selector(arg_lasso, (self.map.nactive + 1,),
                                    rr.affine_transform(self.map.B_active_lasso, offset_active_lasso))

        active_conj_loss_lasso = rr.affine_smooth(self.active_conjugate,_active_lasso)

        cube_obj_lasso = neg_log_cube_probability(self.q_lasso, self.inactive_lagrange, randomization_scale = 1.)

        _inactive_lasso = rr.selector(arg_lasso, (self.map.nactive + 1,),
                                      rr.affine_transform(self.map.B_inactive_lasso, offset_inactive_lasso))

        cube_loss_lasso = rr.affine_smooth(cube_obj_lasso, _inactive_lasso)

        data_simes = self.t * self.map.A_simes

        if self.map.nactive_simes > 1:

            offset_active_simes = self.map.offset_active_simes + data_simes[0]

            _active_simes = rr.selector(arg_simes, (self.map.nactive + 1,),
                                        rr.affine_transform(self.map.B_active_simes, offset_active_simes))

            active_conj_loss_simes = rr.affine_smooth(self.active_conjugate, _active_simes)

            offset_inactive_simes = self.map.offset_inactive_simes + data_simes[1:]

            cube_obj_simes = neg_log_cube_probability(self.q_simes, self.inactive_threshold, randomization_scale=1.)

            _inactive_simes = rr.selector(arg_simes, (self.map.nactive + 1,),
                                          rr.affine_transform(self.map.B_inactive_simes, offset_inactive_simes))

            cube_loss_simes = rr.affine_smooth(cube_obj_simes, _inactive_simes)

            total_loss = rr.smooth_sum([active_conj_loss_lasso,
                                        cube_loss_lasso,
                                        active_conj_loss_simes,
                                        cube_loss_simes,
                                        self.nonnegative_barrier])

        else:

            offset_active_simes = self.map.offset_active_simes + data_simes

            _active_simes = rr.selector(arg_simes, (self.map.nactive + 1,),
                                        rr.affine_transform(self.map.B_active_simes, offset_active_simes))

            active_conj_loss_simes = rr.affine_smooth(self.active_conjugate, _active_simes)

            total_loss = rr.smooth_sum([active_conj_loss_lasso,
                                        cube_loss_lasso,
                                        active_conj_loss_simes,
                                        self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func')
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f, g = total_loss.smooth_objective(param, 'both')
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def minimize2(self, step=1, nstep=30, tol=1.e-6):

        current = self.coefs
        current_value = np.inf

        objective = lambda u: self.sel_prob_smooth_objective(u, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, 'grad')

        for itercount in xrange(nstep):
            newton_step = grad(current)

            # make sure proposal is feasible

            count = 0
            while True:
                count += 1
                proposal = current - step * newton_step
                #print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    #print(proposal)
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                #print("proposal and proposed value", proposal, proposed_value)
                #print(current_value, proposed_value, 'minimize')
                if proposed_value <= current_value:
                    break
                step *= 0.5

            # stop if relative decrease is small

            if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value):
                current = proposal
                current_value = proposed_value
                break

            current = proposal
            current_value = proposed_value

            if itercount % 4 == 0:
                step *= 2

        # print('iter', itercount)
        value = objective(current)
        if value != float('Inf'):
            return current, value
        else:
            raise ValueError("Numerical error")


class approximate_conditional_density_2stage(rr.smooth_atom):

    def __init__(self, sel_alg,
                       coef=1.,
                       offset=None,
                       quadratic=None,
                       nstep=10):

        self.sel_alg = sel_alg

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                coef=coef)

        self.target_observed = self.sel_alg.target_observed
        self.nactive = self.target_observed.shape[0]
        self.target_cov = self.sel_alg.target_cov

    def solve_approx(self):

        #defining the grid on which marginal conditional densities will be evaluated
        grid_length = 361

        #print("observed values", self.target_observed)
        self.ind_obs = np.zeros(self.nactive, int)
        self.norm = np.zeros(self.nactive)
        self.h_approx = np.zeros((self.nactive, grid_length))
        self.grid = np.zeros((self.nactive, grid_length))

        for j in xrange(self.nactive):
            obs = self.target_observed[j]

            self.grid[j,:] = np.linspace(self.target_observed[j]-18., self.target_observed[j]+18.,num=grid_length)

            self.norm[j] = self.target_cov[j,j]
            if obs < self.grid[j,0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid[j,:]):
                self.ind_obs[j] = grid_length-1
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid[j,:]-obs))

            sys.stderr.write("number of variable being computed: " + str(j) + "\n")
            self.h_approx[j, :] = self.approx_conditional_prob(j)
            #print("approx prob", self.h_approx[j, :])

    def approx_conditional_prob(self, j):
        h_hat = []

        self.sel_alg.setup_map(j)

        for i in xrange(self.grid[j,:].shape[0]):
            try:
                approx = approximate_conditional_prob_2stage((self.grid[j,:])[i], self.sel_alg)
                h_hat.append(-(approx.minimize2(nstep=100)[::-1])[0])
            except ValueError:
                if i==0:
                    h_hat.append(0)
                else:
                    h_hat.append(h_hat[i-1])

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.
        approx_nonnormalized = []

        for i in xrange(self.grid[j:,].shape[1]):
            approx_density = np.exp(-np.true_divide(((self.grid[j,:])[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j,:])[i])
            normalizer += approx_density
            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer))

    def approximate_ci(self, j):

        grid_length = 361
        param_grid = np.linspace(-6,12, num=grid_length)
        area = np.zeros(param_grid.shape[0])

        for k in xrange(param_grid.shape[0]):
            area_vec = self.area_normalized_density(j, param_grid[k])
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]
        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0

    def approximate_pvalue(self, j, param):

        area_vec = self.area_normalized_density(j, param)
        area = area_vec[self.ind_obs[j]]

        return 2*min(area, 1-area)

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







