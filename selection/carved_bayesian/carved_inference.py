import numpy as np
import sys

import regreg.api as rr

from selection.carved_bayesian.utils import (nonnegative_softmax_scaled,
                                             neg_log_cube_probability,
                                             projected_langevin,
                                             log_likelihood)

from selection.carved_bayesian.carved_selection_probability import (selection_probability_carved,
                                                                    smooth_cube_barrier)

class sel_inf_carved(rr.smooth_atom):

    def __init__(self, solver, prior_variance, coef=1., offset=None, quadratic=None):

        self.solver = solver

        X, _ = self.solver.loss.data
        self.p_shape = X.shape[1]
        self.param_shape = self.solver._overall.sum()
        self.prior_variance = prior_variance

        initial = self.solver.initial_soln[self.solver._overall]
        print("initial_state", initial)

        rr.smooth_atom.__init__(self,
                                (self.param_shape,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

        self.initial_state = initial

    def smooth_objective_post(self, sel_param, mode='both', check_feasibility=False):

        sel_param = self.apply_offset(sel_param)
        generative_mean = np.zeros(self.p_shape)
        generative_mean[:self.param_shape] = sel_param

        cov_data_inv = self.solver.score_cov_inv

        sel_lasso = selection_probability_carved(self.solver, generative_mean)

        sel_prob_primal = sel_lasso.minimize2(nstep=100)[::-1]

        optimal_primal = (sel_prob_primal[1])[:self.p_shape]

        sel_prob_val = -sel_prob_primal[0]

        full_gradient = cov_data_inv.dot(optimal_primal - generative_mean)

        optimizer = full_gradient[:self.param_shape]

        likelihood_loss = log_likelihood(self.solver.target_observed, self.solver.score_cov[:self.param_shape,
                                                                      :self.param_shape], self.param_shape)

        likelihood_loss_value = likelihood_loss.smooth_objective(sel_param, 'func')

        likelihood_loss_grad = likelihood_loss.smooth_objective(sel_param, 'grad')

        log_prior_loss = rr.signal_approximator(np.zeros(self.param_shape), coef=1. / self.prior_variance)

        log_prior_loss_value = log_prior_loss.smooth_objective(sel_param, 'func')

        log_prior_loss_grad = log_prior_loss.smooth_objective(sel_param, 'grad')

        f = likelihood_loss_value + log_prior_loss_value + sel_prob_val

        g = likelihood_loss_grad + log_prior_loss_grad + optimizer

        if mode == 'func':
            return self.scale(f)
        elif mode == 'grad':
            return self.scale(g)
        elif mode == 'both':
            return self.scale(f), self.scale(g)
        else:
            raise ValueError("mode incorrectly specified")

    def map_solve(self, step=1, nstep=100, tol=1.e-5):

        current = self.coefs[:]
        current_value = np.inf

        objective = lambda u: self.smooth_objective_post(u, 'func')
        grad = lambda u: self.smooth_objective_post(u, 'grad')

        for itercount in range(nstep):

            newton_step = grad(current)

            # make sure proposal is a descent
            count = 0
            while True:
                proposal = current - step * newton_step
                proposed_value = objective(proposal)
                # print("proposal", proposal)

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

    def posterior_samples(self, Langevin_steps=1500, burnin=100):
        state = self.initial_state
        print("here", state.shape)
        gradient_map = lambda x: -self.smooth_objective_post(x, 'grad')
        projection_map = lambda x: x
        stepsize = 1. / (0.5 * self.param_shape)
        sampler = projected_langevin(state, gradient_map, projection_map, stepsize)

        samples = []

        for i in xrange(Langevin_steps):
            sampler.next()
            samples.append(sampler.state.copy())
            sys.stderr.write("sample number: " + str(i) + "\n")

        samples = np.array(samples)
        return samples[burnin:, :]
