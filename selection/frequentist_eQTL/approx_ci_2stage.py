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

            offset_inactive_simes = self.map.offset_inactive_simes + data_simes[self.map.nactive:]

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
