from __future__ import print_function
import sys, os
from math import log
from scipy.stats import norm
import numpy as np
import regreg.api as rr

class fs_selection():

    def __init__(self, randomization, index, T_sign, data_simes, X_unpruned, sigma_ratio):

        self.randomization = randomization
        self.index = index
        self.T_sign = T_sign
        self.data_simes = data_simes
        self.X_unpruned = X_unpruned
        self.sigma_ratio = sigma_ratio
        self.simes_randomization = 0.7
        self.V = self.X_unpruned.shape[1]
        indicator = np.zeros(self.V, dtype=bool)
        indicator[self.index] = 1
        self.perm_X_unpruned = np.hstack([self.X_unpruned[:, indicator], self.X_unpruned[:, ~indicator]])
        self.score_cov_simes = (self.sigma_ratio**2.)* ((self.X_unpruned[:, indicator].T).dot(self.perm_X_unpruned)).T
        self.target_cov = (self.sigma_ratio**2.)*(self.X_unpruned[:, indicator].T.dot(self.X_unpruned[:, indicator]))

        linear_simes = -np.identity(self.V)
        linear_simes[0,0] = -self.T_sign
        self.A_simes = linear_simes.dot(self.score_cov_simes/self.target_cov)
        self.null_statistic_simes = linear_simes.dot(self.data_simes).reshape((self.data_simes.shape[0],1)) \
                                    - self.data_simes[0]* self.A_simes
        self.offset_simes = self.null_statistic_simes

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

class neg_log_cube_probability_fs(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 mu,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.q = q
        self.mu = mu

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)
        self.mu = self.mu.reshape((self.mu.shape[0],))
        arg_u = ((arg *np.ones(self.q)) + self.mu) / self.randomization_scale
        arg_l = (-(arg *np.ones(self.q)) + self.mu) / self.randomization_scale
        prod_arg = np.exp(-(2. * self.mu * (arg *np.ones(self.q))) / (self.randomization_scale ** 2))
        neg_prod_arg = np.exp((2. * self.mu * (arg *np.ones(self.q))) / (self.randomization_scale ** 2))

        cube_prob = norm.cdf(arg_u) - norm.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()

        threshold = 10 ** -8
        indicator = np.zeros(self.q, bool)

        #print("check shape", cube_prob.shape, self.mu.shape, arg_u.shape, arg*np.ones(self.q).shape)

        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(self.mu > 0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)

        log_cube_grad_vec = np.zeros(self.q)
        log_cube_grad_vec[indicator] = -(np.true_divide(norm.pdf(arg_u[indicator]) + norm.pdf(arg_l[indicator]),
                                                    cube_prob[indicator])) / self.randomization_scale

        log_cube_grad_vec[pos_index] = ((1. + prod_arg[pos_index]) /
                                    ((prod_arg[pos_index] / arg_u[pos_index]) +
                                     (1. / arg_l[pos_index]))) / (self.randomization_scale ** 2)

        log_cube_grad_vec[neg_index] = ((arg_u[neg_index] - (arg_l[neg_index] * neg_prod_arg[neg_index]))
                                    / (self.randomization_scale ** 2)) / (1. + neg_prod_arg[neg_index])

        log_cube_grad = log_cube_grad_vec.sum()

        if mode == 'func':
            return self.scale(log_cube_prob).reshape((1,))
        elif mode == 'grad':
            return self.scale(log_cube_grad).reshape((1,))
        elif mode == 'both':
            return self.scale(log_cube_prob).reshape((1,)), self.scale(log_cube_grad).reshape((1,))
        else:
            raise ValueError("mode incorrectly specified")


class log_fs_prob(rr.smooth_atom):

    def __init__(self, t, map, randomization, randomization_scale = 0.7, coef = 1., offset= None, quadratic= None):

        self.t = t
        self.map = map
        data_simes = (self.t * self.map.A_simes)
        self.data_point = self.map.offset_simes + data_simes

        self.randomization_scale = randomization_scale
        self.fs_conjugate = randomization.CGF_conjugate
        self.q = self.data_point.shape[0]-1

        rr.smooth_atom.__init__(self,
                                (1,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=np.ones(1),
                                coef=coef)

        self.coefs[:] = np.ones(1)

        self.nonnegative_barrier = nonnegative_softmax_scaled(1)

    def fs_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        offset_active = self.data_point[0]
        active_conj_loss = rr.affine_smooth(self.fs_conjugate, rr.affine_transform(np.ones(1),
                                                                                   offset_active.reshape((offset_active.shape[0],)))
                                            )
        offset_inactive = self.data_point[1:]
        cube_loss = neg_log_cube_probability_fs(offset_inactive.shape[0],
                                                offset_inactive,
                                                randomization_scale=.7)

        total_loss = rr.smooth_sum([active_conj_loss,
                                    cube_loss,
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

        objective = lambda u: self.fs_prob_smooth_objective(u, 'func')
        grad = lambda u: self.fs_prob_smooth_objective(u, 'grad')

        for itercount in range(nstep):
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

        return current, value


class approximate_conditional_density_2stage(rr.smooth_atom):
    def __init__(self,
                 sel_alg,
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

        self.target_observed = self.sel_alg.data_simes[0]
        self.nactive = 1
        self.target_cov = self.sel_alg.target_cov

    def solve_approx(self):

        # defining the grid on which marginal conditional densities will be evaluated
        grid_length = 301
        self.grid_length = grid_length

        # print("observed values", self.target_observed)
        self.ind_obs = np.zeros(self.nactive, int)
        self.h_approx = np.zeros(grid_length)
        self.grid = np.zeros(grid_length)

        obs = self.target_observed
        self.grid = np.linspace(self.target_observed - 15., self.target_observed + 15., num=grid_length)
        self.norm = self.target_cov

        if obs < self.grid[0]:
            self.ind_obs = 0
        elif obs > np.max(self.grid):
            self.ind_obs = grid_length - 1
        else:
            self.ind_obs = np.argmin(np.abs(self.grid - obs))

        self.h_approx = self.approx_conditional_prob()

    def approx_conditional_prob(self):
        h_hat = []

        #self.sel_alg.setup_map()

        count = 0.
        for i in xrange(self.grid.shape[0]):
            approx = log_fs_prob(self.grid[i], self.sel_alg, self.sel_alg.randomization)
            val = -(approx.minimize2(step=1, nstep=100)[::-1])[0]

            if val != -float('Inf'):
                h_hat.append(val)
                if val< -500.:
                    count +=1
            elif val == -float('Inf') and i == 0:
                h_hat.append(-500.)
                count += 1
            elif val == -float('Inf') and i > 0:
                h_hat.append(h_hat[i - 1])
                count += 1

            #if count > 150:
            #    raise ValueError("Error on grid approx")
            sys.stderr.write("point on grid: " + str(i) + "\n")
            sys.stderr.write("value on grid: " + str(h_hat[i]) + "\n")

        return np.array(h_hat)

    def area_normalized_density(self, mean):

        normalizer = 0.
        approx_nonnormalized = []
        grad_normalizer = 0.

        for i in range(self.grid_length):
            approx_density = np.exp(-np.true_divide((self.grid[i] - mean) ** 2, 2 * self.norm)
                                    + self.h_approx[i])
            normalizer += approx_density
            grad_normalizer += (-mean / self.norm + self.grid[i] / self.norm) * approx_density
            approx_nonnormalized.append(approx_density)

        return np.cumsum(np.array(approx_nonnormalized / normalizer)), normalizer, grad_normalizer

    def smooth_objective_MLE(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        approx_normalizer = self.area_normalized_density(param)

        f = (param ** 2) / (2 * self.norm) - (self.target_observed * param) / self.norm + \
            log(approx_normalizer[1])

        g = param / self.norm - self.target_observed / self.norm + \
            approx_normalizer[2] / approx_normalizer[1]

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def approx_MLE_solver(self, step=1, nstep=150, tol=1.e-5):

        current = self.target_observed
        current_value = np.inf

        objective = lambda u: self.smooth_objective_MLE(u, 'func')
        grad = lambda u: self.smooth_objective_MLE(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current) * self.norm

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

    def approximate_ci(self):

        grid_num = 201
        param_grid = np.linspace(-10, 10, num=grid_num)
        area = np.zeros(param_grid.shape[0])

        for k in xrange(param_grid.shape[0]):
            area_vec = self.area_normalized_density(param_grid[k])[0]
            area[k] = area_vec[self.ind_obs]

        region = param_grid[(area >= 0.05) & (area <= 0.95)]
        if region.size > 0:
            return np.nanmin(region), np.nanmax(region)
        else:
            return 0, 0

    def approximate_pvalue(self, param):

        area_vec = self.area_normalized_density(param)[0]
        area = area_vec[self.ind_obs]

        return 2 * min(area, 1. - area)

def marginal_screening(X_unpruned,
                       index,
                       T_sign,
                       data_simes,
                       T_observed,
                       sigma_ratio=1.):

    from selection.api import randomization

    n, p = X.shape
    randomization = randomization.isotropic_gaussian((p,), scale=.7)

    M_est = fs_selection(randomization, index, T_sign, data_simes, X_unpruned, sigma_ratio)
    nactive = 1
    active_set = index

    truth = 0.
    sys.stderr.write("True target to be covered" + str(truth) + "\n")

    #result = log_fs_prob(1, M_est, randomization)
    #check = result.fs_prob_smooth_objective(-1, 'func')

    ci = approximate_conditional_density_2stage(M_est)
    ci.solve_approx()

    sel_covered = np.zeros(nactive, np.bool)
    pivots = np.zeros(nactive)
    sel_MLE = np.zeros(nactive)
    sel_length = np.zeros(nactive)
    sel_risk = np.zeros(nactive)

    ci_naive = np.asarray([T_observed- 1.65, T_observed + 1.65])
    ci_sel = np.array(ci.approximate_ci())

    print("some details", M_est.target_cov, M_est.data_simes[0], ci_naive, ci_sel)
    pivots = ci.approximate_pvalue(0.)
    sel_MLE = ci.approx_MLE_solver(step=1, nstep=150)[0]
    sel_risk = (sel_MLE - truth) ** 2.
    sel_length = ci_sel[1] - ci_sel[0]
    naive_length = ci_naive[1] - ci_naive[0]
    naive_risk = (T_observed - truth) ** 2.

    if (ci_sel[0] <= truth) and (ci_sel[1] >= truth):
        sel_covered = 1.
    else:
        sel_covered = 0.
    if (ci_naive[0] <= truth) and (ci_naive[1] >= truth):
        naive_covered = 1.
    else:
        naive_covered = 0.

    print("sel covered, naive covered", sel_covered, naive_covered)
    print("sel_risk, naive_risk", sel_risk, naive_risk)
    print("sel_length, naive_length", sel_length, naive_length)
    list_results = np.transpose(np.vstack((ci_sel[0],
                                           ci_sel[1],
                                           ci_naive[0],
                                           ci_naive[1],
                                           pivots,
                                           active_set,
                                           sel_covered,
                                           naive_covered,
                                           sel_risk,
                                           naive_risk,
                                           sel_length,
                                           naive_length)))

    return sel_covered, naive_covered, sel_risk, naive_risk, sel_length, naive_length


if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/sim_Egene_data/'

    gene = str("ENSG00000187642.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    print("shape of X", n, p)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X

    sel_risk = 0.
    naive_risk = 0.
    sel_covered = 0.
    naive_covered = 0.
    sel_length = 0.
    naive_length = 0.

    for seed_n in range(10):

        np.random.seed(seed_n)
        y = np.random.standard_normal(n)
        sigma_est = 1.

        t_test = (X.T.dot(y) + 0.7 * np.random.standard_normal(p)) / np.sqrt(2.)
        index = np.argmax(np.abs(t_test))
        T_sign = np.sign(t_test[index])
        T_observed = t_test[index]
        indicator = np.zeros(p, dtype=bool)
        indicator[index] = 1
        data_simes = np.hstack([X_unpruned[:, indicator].T.dot(y), X_unpruned[:, ~indicator].T.dot(y)])
        results = marginal_screening(X_unpruned,
                                     index,
                                     T_sign,
                                     data_simes,
                                     T_observed)


        sel_covered += results[0]
        naive_covered += results[1]
        sel_risk += results[2]
        naive_risk += results[3]
        sel_length += results[4]
        naive_length += results[5]
        print("iteration completed", seed_n)

    print("results", sel_covered, naive_covered, sel_risk, naive_risk, sel_length, naive_length)



