from __future__ import print_function
import sys
from math import log

import numpy as np
import regreg.api as rr
from selection.randomized.M_estimator import M_estimator
from scipy.stats import norm

class nonnegative_softmax_scaled(rr.smooth_atom):
    """
    The nonnegative softmax objective
    .. math::
         \mu \mapsto
         \sum_{i=1}^{m} \log \left(1 +
         \frac{1}{\mu_i} \right)
    """

    objective_template = r"""\text{nonneg_softmax}\left(%(var)s\right)"""

    def __init__(self,
                 shape,
                 barrier_scale=1.,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        # a feasible point
        self.coefs[:] = np.ones(shape)
        self.barrier_scale = barrier_scale

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        slack = self.apply_offset(mean_param)

        if mode in ['both', 'func']:
            if np.all(slack > 0):
                f = self.scale(np.log((slack + self.barrier_scale) / slack).sum())
            else:
                f = np.inf
        if mode in ['both', 'grad']:
            g = self.scale(1. / (slack + self.barrier_scale) - 1. / slack)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")

class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange)/self.randomization_scale
        arg_l = (arg - self.lagrange)/self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg)/(self.randomization_scale**2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg)/(self.randomization_scale**2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg>0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                        cube_prob[indicator]))/self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index])/
                                     ((prod_arg[pos_index]/arg_u[pos_index])-
                                      (1./arg_l[pos_index])))/self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] -(arg_l[neg_index]*neg_prod_arg[neg_index]))
                                    /self.randomization_scale)/(1.- neg_prod_arg[neg_index])


        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")

class M_estimator_2step(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, simes_level, index, T_sign, l_threshold, u_threshold, data_simes,
                 X_unpruned, sigma_ratio):

        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.simes_level = simes_level
        self.index = index
        self.T_sign = T_sign
        self.data_simes = data_simes
        self.l_threshold = l_threshold
        self.u_threshold = u_threshold
        self.randomization_scale = 1.
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

        self.score_cov_simes = self.sigma_ratio* (X_active_inv.dot(X[:, self._overall].T).dot(self.X_unpruned[:, self.index]))

    def setup_map(self, j):

        self.A_lasso = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic_lasso = self._score_linear_term.dot(self.observed_score_state) - self.A_lasso * self.target_observed[j]

        self.offset_active_lasso = self._opt_affine_term[:self.nactive] + self.null_statistic_lasso[:self.nactive]
        self.offset_inactive_lasso = self.null_statistic_lasso[self.nactive:]

        linear_simes = -self.T_sign
        self.A_simes = linear_simes* (self.score_cov_simes[j] / self.target_cov[j, j])
        self.null_statistic_simes = linear_simes* (self.data_simes) - self.A_simes * self.target_observed[j]

        #print("null_stats", linear_simes* (self.data_simes), self.A_simes * self.target_observed[j], self.null_statistic_simes)
        self.offset_simes = self.null_statistic_simes

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

        self.feasible_point = self.map.feasible_point_lasso

        rr.smooth_atom.__init__(self,
                                (self.map.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.feasible_point,
                                coef=coef)

        self.coefs[:] = self.feasible_point

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.map.nactive)


    def sel_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        data_lasso = np.squeeze(self.t *  self.map.A_lasso)

        offset_active_lasso = self.map.offset_active_lasso + data_lasso[:self.map.nactive]
        offset_inactive_lasso = self.map.offset_inactive_lasso + data_lasso[self.map.nactive:]

        active_conj_loss_lasso = rr.affine_smooth(self.active_conjugate,
                                                  rr.affine_transform(self.map.B_active_lasso, offset_active_lasso))

        cube_obj_lasso = neg_log_cube_probability(self.q_lasso, self.inactive_lagrange, randomization_scale = 1.)

        cube_loss_lasso = rr.affine_smooth(cube_obj_lasso,
                                           rr.affine_transform(self.map.B_inactive_lasso, offset_inactive_lasso))

        data_simes = self.t * self.map.A_simes

        offset_simes = self.map.offset_simes + data_simes

        self.lagrange_2 = self.map.u_threshold
        self.lagrange_1 = self.map.l_threshold

        self.randomization_simes = self.map.simes_randomization

        arg_u = (offset_simes + self.lagrange_2) / self.randomization_simes
        arg_l = (offset_simes + self.lagrange_1) / self.randomization_simes

        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)

        if cube_prob > 10 ** -5:
            log_cube_prob = -np.log(cube_prob).sum()

        elif cube_prob <= 10 ** -5 and offset_simes < 0:
            rand_var = self.randomization_simes ** 2
            log_cube_prob = (offset_simes ** 2. / (2. * rand_var)) + (offset_simes * self.lagrange_2 / rand_var) \
                            - np.log(np.exp(-(self.lagrange_2 ** 2) / (2. * rand_var)) / np.abs(
                (offset_simes + self.lagrange_2) / self.randomization_simes)
                                     - np.exp(
                -(self.lagrange_1 ** 2 + (2. * offset_simes * (self.lagrange_1 - self.lagrange_2))) / (2. * rand_var))
                                     / np.abs((offset_simes + self.lagrange_1) / self.randomization_simes))

        elif cube_prob <= 10 ** -5 and offset_simes > 0:
            rand_var = self.randomization_simes ** 2
            log_cube_prob = (offset_simes ** 2. / (2. * rand_var)) + (offset_simes * self.lagrange_1 / rand_var) \
                            - np.log(
                -np.exp(-(self.lagrange_2 ** 2 + (2. * offset_simes * (self.lagrange_2 - self.lagrange_1)))
                        / (2. * rand_var)) / np.abs((offset_simes + self.lagrange_2) / self.randomization_simes)
                + np.exp(-(self.lagrange_1 ** 2) / (2. * rand_var)) /
                np.abs((offset_simes + self.lagrange_1) / self.randomization_simes))

        total_loss = rr.smooth_sum([active_conj_loss_lasso,
                                    cube_loss_lasso,
                                    self.nonnegative_barrier])

        if mode == 'func':
            f = total_loss.smooth_objective(param, 'func') + log_cube_prob
            return self.scale(f)
        elif mode == 'grad':
            g = total_loss.smooth_objective(param, 'grad')
            return self.scale(g)
        elif mode == 'both':
            f = total_loss.smooth_objective(param, 'func')+ log_cube_prob
            g = total_loss.smooth_objective(param, 'grad')
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
        if value != -float('Inf'):
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
        grid_length = 301

        #print("observed values", self.target_observed)
        self.ind_obs = np.zeros(self.nactive, int)
        self.norm = np.zeros(self.nactive)
        self.h_approx = np.zeros((self.nactive, grid_length))
        self.grid = np.zeros((self.nactive, grid_length))

        for j in xrange(self.nactive):
            obs = self.target_observed[j]

            self.grid[j,:] = np.linspace(self.target_observed[j]-15., self.target_observed[j]+15.,num=grid_length)

            self.norm[j] = self.target_cov[j,j]
            if obs < self.grid[j,0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid[j,:]):
                self.ind_obs[j] = grid_length-1
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid[j,:]-obs))

            sys.stderr.write("number of variable being computed: " + str(j) + "\n")
            self.h_approx[j, :] = self.approx_conditional_prob(j)

    def approx_conditional_prob(self, j):
        h_hat = []

        self.sel_alg.setup_map(j)

        for i in xrange(self.grid[j, :].shape[0]):
            approx = approximate_conditional_prob_2stage((self.grid[j, :])[i], self.sel_alg)
            val = -(approx.minimize2(step=1, nstep=200)[::-1])[0]

            if val != -float('Inf'):
                h_hat.append(val)
            elif val == -float('Inf') and i == 0:
                h_hat.append(-500.)
            elif val == -float('Inf') and i > 0:
                h_hat.append(h_hat[i - 1])
            sys.stderr.write("point on grid: " + str(i) + "\n")
            sys.stderr.write("value on grid: " + str(h_hat[i]) + "\n")

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.
        approx_nonnormalized = []
        grad_normalizer = 0.

        print("shape", self.grid[j:,].shape[1] )
        for i in range(self.grid[j:,].shape[1]):
            approx_density = np.exp(-np.true_divide(((self.grid[j,:])[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j,:])[i])
            normalizer += approx_density
            grad_normalizer += (-mean / self.norm[j] + (self.grid[j, :])[i] / self.norm[j]) * approx_density
            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer)), normalizer, grad_normalizer

    def smooth_objective_MLE(self, param, j, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        approx_normalizer = self.area_normalized_density(j, param)

        f = (param ** 2) / (2 * self.norm[j]) - (self.target_observed[j] * param) / self.norm[j] + \
            log(approx_normalizer[1])

        g = param / self.norm[j] - self.target_observed[j] / self.norm[j] + \
            approx_normalizer[2] / approx_normalizer[1]

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def approx_MLE_solver(self, j, step=1, nstep=150, tol=1.e-5):

        current = self.target_observed[j]
        current_value = np.inf

        objective = lambda u: self.smooth_objective_MLE(u, j, 'func')
        grad = lambda u: self.smooth_objective_MLE(u, j, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current) * self.norm[j]

            # make sure proposal is a descent
            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)

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

        value = objective(current)

        return current, value

    def approximate_ci(self, j):

        grid_length = 301
        param_grid = np.linspace(-10,10, num=grid_length)
        area = np.zeros(param_grid.shape[0])

        for k in xrange(param_grid.shape[0]):
            area_vec = self.area_normalized_density(j, param_grid[k])[0]
            area[k] = area_vec[self.ind_obs[j]]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]
        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0

    def approximate_pvalue(self, j, param):

        area_vec = self.area_normalized_density(j, param)[0]
        area = area_vec[self.ind_obs[j]]

        return 2*min(area, 1.-area)
