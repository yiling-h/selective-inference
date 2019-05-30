from __future__ import print_function
import functools
import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from .query import two_stage_gaussian_query
from .randomization import randomization
from ..base import restricted_estimator

class multi_screening(two_stage_gaussian_query):

    def __init__(self,
                 observed_data,
                 covariance,
                 randomizer,
                 perturb=None):

        self.observed_score_state = -observed_data  # -Z if Z \sim N(\mu,\Sigma), X^Ty in regression setting
        self.nfeature = p = self.observed_score_state.shape[0]
        self.covariance = covariance
        self.randomizer = randomizer
        self._initial_omega = perturb

    def fit(self, perturb=None):
        two_stage_gaussian_query.fit(self, perturb=perturb)
        self._randomized_score = self.observed_score_state - self._initial_omega
        return self._randomized_score, self._randomized_score.shape[0]

    def multivariate_targets(self, features, dispersion=1.):
        """
        Entries of the mean of \Sigma[E,E]^{-1}Z_E
        """
        score_linear = self.covariance[:, features].copy() / dispersion
        Q = score_linear[features]
        cov_target = np.linalg.inv(Q)
        observed_target = -np.linalg.inv(Q).dot(self.observed_score_state[features])
        crosscov_target_score = -score_linear.dot(cov_target)
        alternatives = ['twosided'] * features.sum()

        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    def full_targets(self, features, dispersion=1.):
        """
        Entries of the mean of \Sigma[E,E]^{-1}Z_E
        """
        score_linear = self.covariance[:, features].copy() / dispersion
        Q = self.covariance / dispersion
        cov_target = (np.linalg.inv(Q)[features])[:, features]
        observed_target = -np.linalg.inv(Q).dot(self.observed_score_state)[features]
        crosscov_target_score = -np.identity(Q.shape[0])[:, features]
        alternatives = ['twosided'] * features.sum()

        return observed_target, cov_target * dispersion, crosscov_target_score.T * dispersion, alternatives

    def marginal_targets(self, features):
        """
        Entries of the mean of Z_E
        """
        score_linear = self.covariance[:, features]
        Q = score_linear[features]
        cov_target = Q
        observed_target = -self.observed_score_state[features]
        crosscov_target_score = -score_linear
        alternatives = ['twosided'] * features.sum()

        return observed_target, cov_target, crosscov_target_score.T, alternatives


class marginal_screening(multi_screening):

    useC = True

    def __init__(self,
                 observed_data,
                 covariance,
                 randomizer,
                 threshold,
                 useC=True,
                 perturb=None):

        self.threshold = threshold
        multi_screening.__init__(self,
                                 observed_data,
                                 covariance,
                                 randomizer,
                                 perturb=None)

        self.useC = useC

    def fit(self, perturb=None):

        _randomized_score, p = multi_screening.fit(self, perturb=perturb)
        active = np.fabs(_randomized_score) >= self.threshold

        self._selected = active
        self._not_selected = ~self._selected
        sign = np.sign(-_randomized_score)
        active_signs = sign[self._selected]
        sign[self._not_selected] = 0
        self.selection_variable = {'sign': sign,
                                   'variables': self._selected.copy()}

        self.observed_opt_state = (np.fabs(_randomized_score) - self.threshold)[self._selected]
        self.num_opt_var = self.observed_opt_state.shape[0]

        opt_linear = np.zeros((p, self.num_opt_var))
        opt_linear[self._selected,:] = np.diag(active_signs)
        opt_offset = np.zeros(p)
        opt_offset[self._selected] = active_signs * self.threshold[self._selected]
        opt_offset[self._not_selected] = _randomized_score[self._not_selected]

        self._setup = True

        A_scaling = -np.identity(len(active_signs))
        b_scaling = np.zeros(self.num_opt_var)

        cond_mean, cond_cov, affine_con, logdens_linear, initial_soln = self.set_sampler(A_scaling,
                                                                                         b_scaling,
                                                                                         opt_linear,
                                                                                         opt_offset,
                                                                                         self.useC)

        return self._selected, cond_mean, cond_cov, affine_con, logdens_linear, initial_soln

    @staticmethod
    def type1(observed_data,
              covariance,
              marginal_level,
              randomizer_scale,
              perturb=None,
              useC=False):
        '''
        Threshold
        '''
        randomized_stdev = np.sqrt(np.diag(covariance) + randomizer_scale**2)
        p = covariance.shape[0]
        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        if np.any(perturb) == True:
            threshold = randomized_stdev * ndist.ppf(1. - marginal_level / 2.)
        else:
            stdev = np.sqrt(np.diag(covariance))
            threshold = stdev * ndist.ppf(1. - marginal_level / 2.)

        return marginal_screening(observed_data,
                                  covariance,
                                  randomizer,
                                  threshold,
                                  useC=useC,
                                  perturb=perturb)

class slope(two_stage_gaussian_query):

    def __init__(self,
                 loglike,
                 slope_weights,
                 ridge_term,
                 randomizer,
                 perturb=None):
        r"""
        Create a new post-selection object for the SLOPE problem

        Parameters
        ----------

        loglike : `regreg.smooth.glm.glm`
            A (negative) log-likelihood as implemented in `regreg`.

        slope_weights : np.ndarray
            SLOPE weights for L-1 penalty. If a float,
            it is broadcast to all features.

        ridge_term : float
            How big a ridge term to add?

        randomizer : object
            Randomizer -- contains representation of randomization density.

        perturb : np.ndarray
            Random perturbation subtracted as a linear
            term in the objective function.
        """

        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        if np.asarray(slope_weights).shape == ():
            slope_weights = np.ones(loglike.shape) * slope_weights
        self.slope_weights = np.asarray(slope_weights)

        self.randomizer = randomizer
        self.ridge_term = ridge_term
        self.penalty = rr.slope(slope_weights, lagrange=1.)
        self._initial_omega = perturb  # random perturbation

    def _solve_randomized_problem(self,
                                  perturb=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):
        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term, 0, -self._initial_omega, 0)
        problem = rr.simple_problem(self.loglike, self.penalty)
        initial_soln = problem.solve(quad, **solve_args)
        initial_subgrad = -(self.loglike.smooth_objective(initial_soln, 'grad') +
                            quad.objective(initial_soln, 'grad'))

        return initial_soln, initial_subgrad

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        self.initial_soln, self.initial_subgrad = self._solve_randomized_problem(perturb=perturb, solve_args=solve_args)
        p = self.initial_soln.shape[0]

        # now we have to work out SLOPE details, clusters, etc.

        active_signs = np.sign(self.initial_soln)
        active = self._active = active_signs != 0

        self._overall = overall = active> 0
        self._inactive = inactive = ~self._overall

        _active_signs = active_signs.copy()
        self.selection_variable = {'sign': _active_signs,
                                   'variables': self._overall}


        indices = np.argsort(-np.fabs(self.initial_soln))
        sorted_soln = self.initial_soln[indices]
        initial_scalings = np.sort(np.unique(np.fabs(self.initial_soln[active])))[::-1]
        self.observed_opt_state = initial_scalings
        self._unpenalized = np.zeros(p, np.bool)

        _beta_unpenalized = restricted_estimator(self.loglike, self._overall, solve_args=solve_args)

        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        self.num_opt_var = self.observed_opt_state.shape[0]

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        _hessian_active = np.dot(X.T, X[:, active] * W[:, None])
        _score_linear_term = -_hessian_active
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        cur_indx_array = []
        cur_indx_array.append(0)
        cur_indx = 0
        pointer = 0
        signs_cluster = []
        for j in range(p - 1):
            if np.abs(sorted_soln[j + 1]) != np.abs(sorted_soln[cur_indx]):
                cur_indx_array.append(j + 1)
                cur_indx = j + 1
                sign_vec = np.zeros(p)
                sign_vec[np.arange(j + 1 - cur_indx_array[pointer]) + cur_indx_array[pointer]] = \
                    np.sign(self.initial_soln[indices[np.arange(j + 1 - cur_indx_array[pointer]) + cur_indx_array[pointer]]])
                signs_cluster.append(sign_vec)
                pointer = pointer + 1
                if sorted_soln[j + 1] == 0:
                    break

        signs_cluster = np.asarray(signs_cluster).T

        if signs_cluster.size == 0:
            return active_signs
        else:
            X_clustered = X[:, indices].dot(signs_cluster)
            _opt_linear_term = X.T.dot(X_clustered)

            _, prec = self.randomizer.cov_prec
            opt_linear, opt_offset = (_opt_linear_term, self.initial_subgrad)

            # now make the constraints

            self._setup = True
            A_scaling_0 = -np.identity(self.num_opt_var)
            A_scaling_1 = -np.identity(self.num_opt_var)[:(self.num_opt_var - 1), :]
            for k in range(A_scaling_1.shape[0]):
                A_scaling_1[k, k + 1] = 1
            A_scaling = np.vstack([A_scaling_0, A_scaling_1])
            b_scaling = np.zeros(2 * self.num_opt_var - 1)

            cond_mean, cond_cov, affine_con, logdens_linear, initial_soln =   self._set_sampler(A_scaling,
                                                                                                b_scaling,
                                                                                                opt_linear,
                                                                                                opt_offset)

            return active_signs, cond_mean, cond_cov, affine_con, logdens_linear, initial_soln

    # Targets of inference
    # and covariance with score representation
    # are same as LASSO

    @staticmethod
    def gaussian(X,
                 Y,
                 slope_weights,
                 sigma,
                 quadratic=None,
                 ridge_term=0.,
                 randomizer_scale=None):

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return slope(loglike,
                     np.asarray(slope_weights) / sigma ** 2,
                     ridge_term,
                     randomizer)
