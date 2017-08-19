from __future__ import print_function
import sys
import os
from scipy.stats import norm

import numpy as np
import regreg.api as rr

from selection.randomized.M_estimator import M_estimator
from selection.randomized.query import naive_confidence_intervals
from scipy.stats.stats import pearsonr

from rpy2.robjects.packages import importr
from rpy2 import robjects
glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


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
                 q,  # equals p - E in our case
                 lagrange,
                 randomization_scale=1.,  # equals the randomization variance in our case
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

        arg_u = (arg + self.lagrange) / self.randomization_scale
        arg_l = (arg - self.lagrange) / self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg) / (self.randomization_scale ** 2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg) / (self.randomization_scale ** 2))
        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        #neg_log_cube_prob = -np.log(cube_prob).sum()

        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg > 0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)

        log_cube_prob = np.zeros(self.q)
        log_cube_prob[indicator] = -np.log(cube_prob)[indicator]

        random_var = self.randomization_scale ** 2
        log_cube_prob[neg_index] = (arg[neg_index] ** 2. / (2. * random_var)) + (
        arg[neg_index] * self.lagrange[neg_index] / random_var) + \
                                   (self.lagrange[neg_index] ** 2. / (2. * random_var)) \
                                   - np.log(
            1. / np.abs(arg_u[neg_index]) - neg_prod_arg[neg_index] / np.abs(arg_l[neg_index]))

        log_cube_prob[pos_index] = (arg[pos_index] ** 2. / (2. * random_var)) - (
        arg[pos_index] * self.lagrange[pos_index] / random_var) + \
                                   (self.lagrange[pos_index] ** 2. / (2. * random_var)) \
                                   - np.log(
            1. / np.abs(arg_l[pos_index]) - prod_arg[pos_index] / np.abs(arg_u[pos_index]))

        neg_log_cube_prob = log_cube_prob.sum()

        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                                   cube_prob[indicator])) / self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index]) /
                                    ((prod_arg[pos_index] / np.abs(arg_u[pos_index])) -
                                     (1. / np.abs(arg_l[pos_index])))) / self.randomization_scale

        log_cube_grad[neg_index] = ((-1. + neg_prod_arg[neg_index]) /
                                    ((-neg_prod_arg[neg_index] / np.abs(arg_l[neg_index])) +
                                     (1. / np.abs(arg_u[neg_index])))) / self.randomization_scale

        if mode == 'func':
            return self.scale(neg_log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(neg_log_cube_prob), self.scale(log_cube_grad)
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

        self.offset_simes = self.null_statistic_simes


class approximate_conditional_prob_2stage(rr.smooth_atom):
    def __init__(self,
                 t,  # point at which density is to computed
                 map,
                 coef=1.,
                 offset=None,
                 quadratic=None):

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

        data_lasso = np.squeeze(self.t * self.map.A_lasso)

        offset_active_lasso = self.map.offset_active_lasso + data_lasso[:self.map.nactive]
        offset_inactive_lasso = self.map.offset_inactive_lasso + data_lasso[self.map.nactive:]

        active_conj_loss_lasso = rr.affine_smooth(self.active_conjugate,
                                                  rr.affine_transform(self.map.B_active_lasso, offset_active_lasso))

        cube_obj_lasso = neg_log_cube_probability(self.q_lasso, self.inactive_lagrange, randomization_scale=.7)

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

        if cube_prob > 10 ** -3:
            log_cube_prob = -np.log(cube_prob).sum()

        elif cube_prob <= 10 ** -3 and offset_simes < 0:
            rand_var = self.randomization_simes ** 2
            log_cube_prob = (offset_simes ** 2. / (2. * rand_var)) + (offset_simes * self.lagrange_2 / rand_var) \
                            - np.log(np.exp(-(self.lagrange_2 ** 2) / (2. * rand_var)) / np.abs(
                (offset_simes + self.lagrange_2) / self.randomization_simes)
                                     - np.exp(
                -(self.lagrange_1 ** 2 + (2. * offset_simes * (self.lagrange_1 - self.lagrange_2))) / (2. * rand_var))
                                     / np.abs((offset_simes + self.lagrange_1) / self.randomization_simes))

        elif cube_prob <= 10 ** -3 and offset_simes > 0:
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
            f = total_loss.smooth_objective(param, 'func') + log_cube_prob
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
                # print("current proposal and grad", proposal, newton_step)
                if np.all(proposal > 0):
                    break
                step *= 0.5
                if count >= 40:
                    # print(proposal)
                    raise ValueError('not finding a feasible point')

            # make sure proposal is a descent

            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print("proposal and proposed value", proposal, proposed_value)
                # print(current_value, proposed_value, 'minimize')
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

        # defining the grid on which marginal conditional densities will be evaluated
        grid_length = 301
        self.grid_length = grid_length

        # print("observed values", self.target_observed)
        self.ind_obs = np.zeros(self.nactive, int)
        self.norm = np.zeros(self.nactive)
        self.h_approx = np.zeros((self.nactive, grid_length))
        self.grid = np.zeros((self.nactive, grid_length))

        for j in xrange(self.nactive):
            obs = self.target_observed[j]

            self.grid[j, :] = np.linspace(self.target_observed[j] - 15., self.target_observed[j] + 15., num=grid_length)

            self.norm[j] = self.target_cov[j, j]
            if obs < self.grid[j, 0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid[j, :]):
                self.ind_obs[j] = grid_length - 1
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid[j, :] - obs))

            sys.stderr.write("number of variable being computed: " + str(j) + "\n")
            self.h_approx[j, :] = self.approx_conditional_prob(j)

    def approx_conditional_prob(self, j):
        h_hat = []

        self.sel_alg.setup_map(j)

        count = 0.
        for i in xrange(self.grid[j, :].shape[0]):
            approx = approximate_conditional_prob_2stage((self.grid[j, :])[i], self.sel_alg)
            val = -(approx.minimize2(step=1, nstep=200)[::-1])[0]

            if val != -float('Inf'):
                h_hat.append(val)
            elif val == -float('Inf') and i == 0:
                h_hat.append(-500.)
                count += 1
            elif val == -float('Inf') and i > 0:
                h_hat.append(h_hat[i - 1])
                count += 1

            if count > 150:
                raise ValueError("Error on grid approx")
            sys.stderr.write("point on grid: " + str(i) + "\n")
            sys.stderr.write("value on grid: " + str(h_hat[i]) + "\n")

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.
        approx_nonnormalized = []
        grad_normalizer = 0.

        for i in range(self.grid_length):
            approx_density = np.exp(-np.true_divide(((self.grid[j, :])[i] - mean) ** 2, 2 * self.norm[j])
                                    + (self.h_approx[j, :])[i])
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

        grid_num = 301
        param_grid = np.linspace(-10, 10, num=grid_num)
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

        return 2 * min(area, 1. - area)

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
                             sigma,
                             simes_level,
                             index,
                             T_sign,
                             l_threshold,
                             u_threshold,
                             data_simes,
                             X_unpruned,
                             sigma_ratio,
                             seed_n = 0,
                             bh_level = 0.10,
                             lam_frac = 1.2,
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

    if nactive == 0:
        return None

    else:

        ci = approximate_conditional_density_2stage(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        pivots = np.zeros(nactive)
        sel_MLE = np.zeros(nactive)

        for j in xrange(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            pivots[j] = ci.approximate_pvalue(j, 0.)
            sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]

        sel_length = (ci_sel[:, 1] - ci_sel[:, 0]).sum() / nactive

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = (ci_naive[:,1]- ci_naive[:,0]).sum()/nactive

        p_BH = BH_q(pivots, bh_level)

        discoveries_active = np.zeros(nactive)
        if p_BH is not None:
            for indx in p_BH[1]:
                discoveries_active[indx] = 1

        print("lengths", sel_length, naive_length)
        print("selective intervals", ci_sel.T)
        print("selective MLE", sel_MLE)
        print("uandjusted MLE", M_est.target_observed)
        print("naive intervals", ci_naive.T)

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

    prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])
    X = X[:, prototypes]

    y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
    y = y.reshape((y.shape[0],))

    sigma_est = glmnet_sigma(X, y)
    print("sigma est", sigma_est)

    y /= sigma_est

    simes_output = np.loadtxt(os.path.join(inpath + "simes_" + gene) + ".txt")

    simes_level = (0.10 * 2195)/21819.
    index = int(simes_output[2])
    T_sign = simes_output[4]

    V = simes_output[0]
    u = simes_output[3]
    sigma_hat = simes_output[5]

    if u > 10 ** -12.:
        l_threshold = np.sqrt(1+ (0.7**2)) * norm.ppf(1. - min(u, simes_level * (1./ V)) / 2.)
    else:
        l_threshold = np.sqrt(1 + (0.7 ** 2)) * norm.ppf(1. -(simes_level * (1./ V)/2.))

    u_threshold = 10 ** 10

    data_simes = (sigma_est/sigma_hat)*(X_unpruned[:, index].T.dot(y))

    sigma = 1.

    ratio = sigma_est/sigma_hat

    try:
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
                                           ratio,
                                           seed_n=0)

    except ValueError:
        sys.stderr.write("Value error: error try again!" + "\n")
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
                                           ratio,
                                           seed_n=1)

    outfile = os.path.join(outdir + "inference_" + gene + ".txt")
    np.savetxt(outfile, results)