import numpy as np

import regreg.api as rr
from .query import gaussian_query
from .randomization import randomization
from ..base import restricted_estimator

class multi_task_lasso(gaussian_query):

     def __init__(self,
                  loglikes,
                  feature_weight,
                  ridge_terms,
                  randomizers,
                  nfeature,
                  ntask,
                  perturbations=None):

         self.loglikes = loglikes

         self.ntask = ntask
         self.nfeature = nfeature

         self.ridge_terms = ridge_terms
         self.feature_weight = feature_weight

         self._initial_omega = perturbations
         self.randomizers = randomizers


     def fit(self,
             perturbations=None,
             solve_args={'tol': 1.e-12, 'min_its': 50}):

         p = self.nfeature

         (self.initial_solns,
          self.initial_subgrads) = self._solve_multitasking_problem(perturbations=perturbations)

         active_signs = np.array([np.sign(self.initial_solns[:, i]) for i in range(self.ntask)]).T
         active = self._active = np.array([(active_signs[:, i] != 0) for i in range(self.ntask)]).T

         self._overall = {i: (active[:, i] > 0) for i in range(self.ntask)}
         self._inactive = inactive = {i: ~self._overall[i] for i in range(self.ntask)}
         _seltasks = np.array([active[j,:].sum() for j in range(p)])
         self._seltasks = _seltasks[_seltasks>0]
         print("number of selected tasks across variables ", self._seltasks)

         _active_signs = active_signs.copy()
         ordered_variables = {i: np.nonzero(active[:, i])[0] for i in range(self.ntask)}

         self.selection_variable = {'sign': _active_signs,
                                    'variables': ordered_variables}

         initial_scalings = {i: np.fabs((self.initial_solns[:, i])[active[:, i]]) for i in range(self.ntask)}
         self.observed_opt_state = initial_scalings

         _beta_unpenalized = {i: restricted_estimator(self.loglikes[i],
                                                      self._overall[i],
                                                      solve_args=solve_args) for i in range(self.ntask)}

         def signed_basis_vector(p, j, s):
             v = np.zeros(p)
             v[j] = s
             return v

         beta_bar = np.zeros((p, self.ntask))
         num_opt_var = np.array([self.observed_opt_state[i].shape[0] for i in range(self.ntask)])

         _score_linear_term = {}
         observed_score_state = {}
         opt_linear = {}
         opt_offset = {i: self.initial_subgrads[:, i] for i in range(self.ntask)}

         for j in range(self.ntask):
             beta_bar[self._overall[j], j] = _beta_unpenalized[j]

             X, y = self.loglikes[j].data
             W = self._W = self.loglikes[j].saturated_loss.hessian(X.dot(beta_bar[:,j]))

             _hessian_active = np.dot(X.T, X[:, active[:, j]] * W[:, None])
             _score_linear_term[j] = _hessian_active

             _observed_score_state = _hessian_active.dot(_beta_unpenalized[j])
             _observed_score_state[inactive[j]] += self.loglikes[j].smooth_objective(beta_bar[:,j], 'grad')[inactive[j]]
             observed_score_state[j] = _observed_score_state

             active_directions = np.array([signed_basis_vector(p,
                                                               k,
                                                               (active_signs[:,j])[k])
                                           for k in np.nonzero(active[:,j])[0]]).T

             _opt_linear = np.zeros((p, num_opt_var[j]))
             scaling_slice = slice(0, active[:,j].sum())
             if np.sum(active[:,j]) == 0:
                 _opt_hessian = 0
             else:
                 _opt_hessian = (_hessian_active * (active_signs[:,j])[None, active[:,j]]
                                 + self.ridge_terms[j] * active_directions)

             _opt_linear[:, scaling_slice] = _opt_hessian
             opt_linear[j] = _opt_linear

         self._beta_full = beta_bar

         ###permuting the optimization variables
         ##Pi o = o'

         Pi = np.zeros(num_opt_var.sum())
         indx = 0
         for irow, icol in np.ndindex(active.shape):
             if active[irow, icol] > 0:
                 Pi[indx] = active[:, :(icol+1)].sum()- active[irow:, icol].sum()
                 indx += 1
         Pi= Pi.astype(int)

         print("check permuted indices ", Pi, num_opt_var.sum())

         for k in range(self.ntask):
             X, y = self.loglikes[k].data
             print("check  K.K.T. map", np.allclose(self._initial_omega[:, k], -X.T.dot(y) + opt_offset[k] + opt_linear[k].dot(self.observed_opt_state[k])))

         return active_signs

     def _solve_randomized_problem(self,
                                   penalty,
                                   solve_args={'tol': 1.e-12, 'min_its': 50}):

        quad_list = [rr.identity_quadratic(self.ridge_terms[i],
                                           0,
                                           -self._initial_omega[:, i],
                                           0)
                     for i in range(self.ntask)]

        problem_list = [rr.simple_problem(self.loglikes[i], penalty) for i in range(self.ntask)]

        initial_solns = np.array([problem_list[i].solve(quad_list[i], **solve_args) for i in range(self.ntask)])
        initial_subgrads = np.array([-(self.loglikes[i].smooth_objective(initial_solns[i, :],
                                                                        'grad') +
                                       quad_list[i].objective(initial_solns[i, :], 'grad'))
                                     for i in range(self.ntask)])

        return initial_solns, initial_subgrads

     def _solve_multitasking_problem(self, perturbations=None, num_iter=100, atol=1.e-5):

        if perturbations is not None:
            self._initial_omega = perturbations

        if self._initial_omega is None:
            self._initial_omega = np.array([self.randomizers[i].sample() for i in range(self.ntask)]).T

        penalty_init = rr.weighted_l1norm(self.feature_weight, lagrange=1.)
        solution_init = self._solve_randomized_problem(penalty= penalty_init)
        beta = solution_init[0].T

        for itercount in range(num_iter):

            beta_prev = beta.copy()
            sum_all_tasks = np.sum(np.absolute(beta), axis=1)

            penalty_weight = 1. / np.maximum(np.sqrt(sum_all_tasks), 10 ** -10)

            feature_weight_current = self.feature_weight * penalty_weight

            penalty_current = rr.weighted_l1norm(feature_weight_current, lagrange=1.)

            solution_current = self._solve_randomized_problem(penalty=penalty_current)

            beta = solution_current[0].T

            if np.sum(np.fabs(beta_prev[0].T - beta)) < atol:
                break

        print("check itercount ", itercount)
        subgrad = solution_current[1].T

        return beta, subgrad


     @staticmethod
     def gaussian(predictor_vars,
                 response_vars,
                 feature_weight,
                 noise_levels=None,
                 quadratic=None,
                 ridge_term=None,
                 randomizer_scales=None):

        ntask = len(response_vars)

        if noise_levels is None:
            noise_levels = np.ones(ntask)

        loglikes = {i: rr.glm.gaussian(predictor_vars[i], response_vars[i], coef=1. / noise_levels[i] ** 2, quadratic=quadratic) for i in range(ntask)}

        sample_sizes = np.asarray([predictor_vars[i].shape[0] for i in range(ntask)])
        nfeatures = [predictor_vars[i].shape[1] for i in range(ntask)]

        if all(x == nfeatures[0] for x in nfeatures) == False:
            raise ValueError("all the predictor matrices must have the same regression dimensions")
        else:
            nfeature = nfeatures[0]

        if ridge_term is None:
            ridge_terms = np.zeros(ntask)

        mean_diag_list = [np.mean((predictor_vars[i] ** 2).sum(0)) for i in range(ntask)]
        if randomizer_scales is None:
                randomizer_scales = np.asarray([np.sqrt(mean_diag_list[i]) * 0.5 * np.std(response_vars[i])
                                                * np.sqrt(sample_sizes[i] / (sample_sizes[i] - 1.)) for i in range(ntask)])

        randomizers = {i: randomization.isotropic_gaussian((nfeature,), randomizer_scales[i]) for i in range(ntask)}

        return multi_task_lasso(loglikes,
                                np.asarray(feature_weight),
                                ridge_terms,
                                randomizers,
                                nfeature,
                                ntask)




