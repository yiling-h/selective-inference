from __future__ import print_function
from math import log
import sys, os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr
from selection.randomized.query import naive_confidence_intervals
from selection.tests.instance import gaussian_instance
from selection.randomized.M_estimator import M_estimator

class M_estimator_map(M_estimator):

    def __init__(self, loss, epsilon, penalty, randomization, randomization_scale = 0.7):
        M_estimator.__init__(self, loss, epsilon, penalty, randomization)
        self.randomization_scale = randomization_scale

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
        self.feasible_point = np.abs(self.initial_soln[self._overall])
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

        self.B_active = self._opt_linear_term[:nactive, :nactive]
        self.B_inactive = self._opt_linear_term[nactive:, :nactive]


    def setup_map(self, j):

        self.A = np.dot(self._score_linear_term, self.score_target_cov[:, j]) / self.target_cov[j, j]
        self.null_statistic = self._score_linear_term.dot(self.observed_score_state) - self.A * self.target_observed[j]

        self.offset_active = self._opt_affine_term[:self.nactive] + self.null_statistic[:self.nactive]
        self.offset_inactive = self.null_statistic[self.nactive:]


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
        cube_prob = normal.cdf(arg_u) - normal.cdf(arg_l)

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
        log_cube_grad[indicator] = (np.true_divide(-normal.pdf(arg_u[indicator]) + normal.pdf(arg_l[indicator]),
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

class approximate_conditional_prob(rr.smooth_atom):

    def __init__(self,
                 t, #point at which density is to computed
                 map,
                 coef = 1.,
                 offset= None,
                 quadratic= None):

        self.t = t
        self.map = map
        self.q = map.p - map.nactive
        self.inactive_conjugate = self.active_conjugate = map.randomization.CGF_conjugate

        if self.active_conjugate is None:
            raise ValueError(
                'randomization must know its CGF_conjugate -- currently only isotropic_gaussian and laplace are implemented and are assumed to be randomization with IID coordinates')

        self.inactive_lagrange = self.map.inactive_lagrange

        rr.smooth_atom.__init__(self,
                                (map.nactive,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=self.map.feasible_point,
                                coef=coef)

        self.coefs[:] = map.feasible_point

        self.nonnegative_barrier = nonnegative_softmax_scaled(self.map.nactive)


    def sel_prob_smooth_objective(self, param, mode='both', check_feasibility=False):

        param = self.apply_offset(param)

        data = np.squeeze(self.t *  self.map.A)

        offset_active = self.map.offset_active + data[:self.map.nactive]
        offset_inactive = self.map.offset_inactive + data[self.map.nactive:]

        active_conj_loss = rr.affine_smooth(self.active_conjugate,
                                            rr.affine_transform(self.map.B_active, offset_active))


        cube_obj = neg_log_cube_probability(self.q, self.inactive_lagrange, randomization_scale = self.map.randomization_scale)

        cube_loss = rr.affine_smooth(cube_obj, rr.affine_transform(self.map.B_inactive, offset_inactive))

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

        objective = lambda u: self.sel_prob_smooth_objective(u, 'func')
        grad = lambda u: self.sel_prob_smooth_objective(u, 'grad')

        for itercount in range(nstep):
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

        return current, value

class approximate_conditional_density(rr.smooth_atom):

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
        self.grid_length = 241

        #print("observed values", self.target_observed)
        self.ind_obs = np.zeros(self.nactive, int)
        self.norm = np.zeros(self.nactive)
        self.h_approx = np.zeros((self.nactive, self.grid_length))
        self.grid = np.zeros((self.nactive, self.grid_length))

        for j in range(self.nactive):
            obs = self.target_observed[j]

            self.grid[j,:] = np.linspace(self.target_observed[j]-12., self.target_observed[j]+12.,num=self.grid_length)

            self.norm[j] = self.target_cov[j,j]
            if obs < self.grid[j,0]:
                self.ind_obs[j] = 0
            elif obs > np.max(self.grid[j,:]):
                self.ind_obs[j] = self.grid_length-1
            else:
                self.ind_obs[j] = np.argmin(np.abs(self.grid[j,:]-obs))

            sys.stderr.write("number of variable being computed: " + str(j) + "\n")
            self.h_approx[j, :] = self.approx_conditional_prob(j)

    def approx_conditional_prob(self, j):
        h_hat = []

        self.sel_alg.setup_map(j)

        for i in range(self.grid[j, :].shape[0]):
            approx = approximate_conditional_prob((self.grid[j, :])[i], self.sel_alg)
            val = -(approx.minimize2(step=1, nstep=200)[::-1])[0]

            if val != -float('Inf'):
                h_hat.append(val)
            elif val == -float('Inf') and i == 0:
                h_hat.append(-500.)
            elif val == -float('Inf') and i > 0:
                h_hat.append(h_hat[i - 1])

            #sys.stderr.write("point on grid: " + str(i) + "\n")
            #sys.stderr.write("value on grid: " + str(h_hat[i]) + "\n")

        return np.array(h_hat)

    def area_normalized_density(self, j, mean):

        normalizer = 0.
        approx_nonnormalized = []
        grad_normalizer = 0.

        for i in range(self.grid_length):
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

        grid_num = 301
        param_grid = np.linspace(-10,10, num=grid_num)
        area = np.zeros(param_grid.shape[0])

        for k in range(param_grid.shape[0]):
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


def test_approximate_inference(X,
                               y,
                               true_mean,
                               sigma=1.,
                               seed_n = 0,
                               lam_frac = 1.,
                               loss='gaussian',
                               randomization_scale = 0.7):

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

    randomization = randomization.isotropic_gaussian((p,), scale=randomization_scale)
    M_est = M_estimator_map(loss, epsilon, penalty, randomization, randomization_scale = randomization_scale)

    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)
    sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
    sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
    sys.stderr.write("Observed target" + str(M_est.target_observed) + "\n")

    if nactive == 0:
        return 0, 0, 0, 0, 0, 0

    else:
        true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(true_mean)

        sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

        class target_class(object):
            def __init__(self, target_cov):
                self.target_cov = target_cov
                self.shape = target_cov.shape

        target = target_class(M_est.target_cov)

        ci_naive = naive_confidence_intervals(target, M_est.target_observed)
        naive_covered = np.zeros(nactive, np.bool)
        naive_length = (ci_naive[:,1]-ci_naive[:,0]).sum()/nactive
        naive_risk = ((M_est.target_observed-true_vec)**2.).sum()/nactive

        ci = approximate_conditional_density(M_est)
        ci.solve_approx()

        ci_sel = np.zeros((nactive, 2))
        sel_MLE = np.zeros(nactive)
        sel_length = np.zeros(nactive)

        sel_covered = np.zeros(nactive, np.bool)
        sel_risk = np.zeros(nactive)

        for j in range(nactive):
            ci_sel[j, :] = np.array(ci.approximate_ci(j))
            if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
                sel_covered[j] = 1
            if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
                naive_covered[j] = 1
            sel_MLE[j] = ci.approx_MLE_solver(j, step=1, nstep=150)[0]
            sel_risk[j] = (sel_MLE[j] - true_vec[j]) ** 2.
            sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]

        #print("lengths", sel_length.sum()/nactive)
        #print("selective intervals", ci_sel.T)
        #print("risks", sel_risk.sum() / nactive)

        # return np.transpose(np.vstack((ci_sel[:, 0],
        #                                ci_sel[:, 1],
        #                                sel_MLE,
        #                                sel_covered,
        #                                sel_risk)))

        return sel_covered.sum()/float(nactive), naive_covered.sum()/float(nactive), sel_risk.sum() / nactive, \
               naive_risk, sel_length.sum() / nactive, naive_length

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/sim_Egene_data/'

    gene = str("ENSG00000187642.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X

    prototypes = np.loadtxt(os.path.join(path + "protoclust_" + gene) + ".txt", delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])
    X = X[:, prototypes]
    n, p = X.shape
    print("shape of X", n, p)

    sel_risk = 0.
    naive_risk = 0.
    sel_covered = 0.
    naive_covered = 0.
    sel_length = 0.
    naive_length = 0.
    count = 0
    for seed_n in range(10):

        np.random.seed(seed_n+90)
        y = np.random.standard_normal(n)
        true_mean = np.zeros(n)
        try:
            results = test_approximate_inference(X,
                                                 y,
                                                 true_mean)

            print("results", results)
            if results[4] == 0 and results[5] == 0:
                count += 1
            else:
                sel_covered += results[0]
                naive_covered += results[1]
                sel_risk += results[2]
                naive_risk += results[3]
                sel_length += results[4]
                naive_length += results[5]
                print("iteration completed", seed_n)
        except:
            count += 1

    print("results", count, sel_covered, naive_covered, sel_risk, naive_risk, sel_length, naive_length)