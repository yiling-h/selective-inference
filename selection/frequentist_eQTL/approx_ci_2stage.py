import numpy as np
import sys
import regreg.api as rr
from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from scipy.stats import norm

from selection.frequentist_eQTL.approx_confidence_intervals import neg_log_cube_probability

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