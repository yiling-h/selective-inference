from __future__ import print_function
from scipy.linalg import block_diag
from scipy.stats import norm as ndist
from scipy.interpolate import interp1d

import collections
import numpy as np
from numpy import log
from numpy.linalg import norm, qr, inv, eig
import pandas as pd

import regreg.api as rr
from .randomization import randomization
from ..base import restricted_estimator
from ..algorithms.barrier_affine import solve_barrier_affine_py as solver
from ..distributions.discrete_family import discrete_family

class nbhd_lasso(object):

    def __init__(self,
                 X,
                 #loglike,
                 #groups,
                 weights,
                 ridge_term,
                 randomizer,
                 perturb=None):

        # log likelihood : quadratic loss
        # self.loglike = loglike
        self.nfeature = X.shape[1]
        if np.asarray(weights).shape == ():
            weights = np.ones((self.nfeature,self.nfeature - 1)) * weights
            print(weights.shape)
            print(weights)
        self.weights = np.asarray(weights)
        print(weights.shape)
        print("weights[5]:",weights[5])

        # ridge parameter
        self.ridge_term = ridge_term

        self.penalty = []
        # a for loop
        # self.penalty = rr.weighted_l1norm(self.feature_weights, lagrange=1.)
        for i in range(self.nfeature):
            self.penalty.append(rr.weighted_l1norm(self.weights[i], lagrange=1.))

        self._initial_omega = perturb

        # gaussian randomization
        self.randomizer = randomizer

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):
        """

        Fit the randomized lasso using `regreg`.

        Parameters
        ----------

        solve_args : keyword args
             Passed to `regreg.problems.simple_problem.solve`.

        Returns
        -------

        signs : np.float
             Support and non-zero signs of randomized lasso solution.

        """

        p = self.nfeature

        (self.observed_soln,
         self.observed_subgrad) = self._solve_randomized_problem(
            perturb=perturb,
            solve_args=solve_args)

        active_signs = np.sign(self.observed_soln)
        active = self._active = active_signs != 0  # flag for nonzero coeffs

        self._lagrange = self.penalty.weights
        unpenalized = self._lagrange == 0

        active *= ~unpenalized  # flag for nonzero AND penalized coeffs

        self._overall = overall = (active + unpenalized) > 0
        self._inactive = inactive = ~self._overall
        self._unpenalized = unpenalized

        _active_signs = active_signs.copy()

        # don't release sign of unpenalized variables
        _active_signs[unpenalized] = np.nan
        ordered_variables = list((tuple(np.nonzero(active)[0]) +
                                  tuple(np.nonzero(unpenalized)[0])))
        self.selection_variable = {'sign': _active_signs,
                                   'variables': ordered_variables}

        # initial state for opt variables

        initial_scalings = np.fabs(self.observed_soln[active])
        initial_unpenalized = self.observed_soln[self._unpenalized]

        # Abs. values of active vars & original values of unpenalized vars
        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized])
        # For LASSO, this is the OLS solution on X_{E,U}
        _beta_unpenalized = restricted_estimator(self.loglike,
                                                 self._overall,
                                                 solve_args=solve_args)

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator
        # beta_bar: restricted OLS solution + some zeros at appropriate positions
        beta_bar = np.zeros(p)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        # form linear part

        num_opt_var = self.observed_opt_state.shape[0]

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        # compute part of hessian
        # _hessian: X'X, _hessian_active: X'X_E, _hessian_unpen: X'X_U
        _hessian, _hessian_active, _hessian_unpen = _compute_hessian(self.loglike,
                                                                     beta_bar,
                                                                     active,
                                                                     unpenalized)

        # fill in pieces of query

        opt_linear = np.zeros((p, num_opt_var))
        _score_linear_term = np.zeros((p, num_opt_var))

        _score_linear_term = -np.hstack([_hessian_active, _hessian_unpen])  # - X'X_{E,U}

        # set the observed score (data dependent) state

        # observed_score_state is
        # \nabla \ell(\bar{\beta}_E) - Q(\bar{\beta}_E) \bar{\beta}_E
        # in linear regression this is _ALWAYS_ -X^TY
        #
        # should be asymptotically equivalent to
        # \nabla \ell(\beta^*) - Q(\beta^*)\beta^*

        self.observed_score_state = _score_linear_term.dot(_beta_unpenalized)
        self.observed_score_state[inactive] += self.loglike.smooth_objective(beta_bar, 'grad')[inactive]

        def signed_basis_vector(p, j, s):
            v = np.zeros(p)
            v[j] = s
            return v

        active_directions = np.array([signed_basis_vector(p,
                                                          j,
                                                          active_signs[j])
                                      for j in np.nonzero(active)[0]]).T

        scaling_slice = slice(0, active.sum())
        if np.sum(active) == 0:
            _opt_hessian = 0
        else:
            _opt_hessian = (_hessian_active * active_signs[None, active]
                            + self.ridge_term * active_directions)

        opt_linear[:, scaling_slice] = _opt_hessian

        # beta_U piece

        unpenalized_slice = slice(active.sum(), num_opt_var)
        unpenalized_directions = np.array([signed_basis_vector(p, j, 1) for
                                           j in np.nonzero(unpenalized)[0]]).T
        if unpenalized.sum():
            opt_linear[:, unpenalized_slice] = (_hessian_unpen
                                                + self.ridge_term *
                                                unpenalized_directions)

        self.opt_linear = opt_linear
        # now make the constraints and implied gaussian

        self._setup = True
        A_scaling = -np.identity(num_opt_var)
        b_scaling = np.zeros(num_opt_var)

        #### to be fixed -- set the cov_score here without dispersion

        self._unscaled_cov_score = _hessian

        self.num_opt_var = num_opt_var

        self._setup_sampler_data = (A_scaling[:active.sum()],
                                    b_scaling[:active.sum()],
                                    opt_linear,
                                    self.observed_subgrad)

        return active_signs

    def setup_inference(self,
                        dispersion):

        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data,
                                dispersion=dispersion)

    def _solve_randomized_problem(self,
                                  perturb=None,
                                  solve_args={'tol': 1.e-12, 'min_its': 50}):

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

        quad = rr.identity_quadratic(self.ridge_term,
                                     0,
                                     -self._initial_omega,
                                     0)

        problem = rr.simple_problem(self.loglike, self.penalty)

        observed_soln = problem.solve(quad, **solve_args)
        observed_subgrad = -(self.loglike.smooth_objective(observed_soln,
                                                           'grad') +
                             quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad