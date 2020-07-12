import numpy as np

import regreg.api as rr
from .query import gaussian_query
from .randomization import randomization

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

    def _solve_randomized_problem(self,
                                  penalty,
                                  perturbations=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        if perturbations is not None:
            self._initial_omega = perturbations

        if self._initial_omega is None:
            self._initial_omega = np.array([self.randomizers[i].sample() for i in range(self.ntask)]).T

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

    def multitasking_solver(self, num_iter=50):

        penalty_prev = rr.weighted_l1norm(self.feature_weight, lagrange=1.)
        solution_prev = self._solve_randomized_problem(penalty= penalty_prev)
        beta = solution_prev[0].T

        for iteration in range(num_iter):

            sum_all_tasks = np.sum(np.absolute(beta), axis=1)
            penalty_weight = 1. / np.maximum(np.sqrt(sum_all_tasks), 10 ** -10)

            feature_weight_current = self.feature_weight * penalty_weight

            penalty_current = rr.weighted_l1norm(feature_weight_current, lagrange=1.)

            solution_current = self._solve_randomized_problem(penalty=penalty_current)

            beta = solution_current[0].T

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




