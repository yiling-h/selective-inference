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
from .query_jacobian import gaussian_query
from ..base import (restricted_estimator,
                    _compute_hessian,
                    _pearsonX2)
from ..algorithms.barrier_affine import solve_barrier_affine_py as solver
from ..distributions.discrete_family import discrete_family

class group_lasso(gaussian_query):

    def __init__(self,
                 loglike,
                 groups,
                 weights,
                 ridge_term,
                 randomizer,
                 useJacobian=True,
                 use_lasso=True,  # should lasso solver be used where applicable - defaults to True
                 perturb=None):

        _check_groups(groups)  # make sure groups looks sensible

        # log likelihood : quadratic loss
        self.loglike = loglike
        self.nfeature = self.loglike.shape[0]

        # ridge parameter
        self.ridge_term = ridge_term

        # group lasso penalty (from regreg)
        # use regular lasso penalty if all groups are size 1
        if use_lasso and groups.size == np.unique(groups).size:
            # need to provide weights an an np.array rather than a dictionary
            weights_np = np.array([w[1] for w in sorted(weights.items())])
            self.penalty = rr.weighted_l1norm(weights=weights_np,
                                              lagrange=1.)
        else:
            self.penalty = rr.group_lasso(groups,
                                          weights=weights,
                                          lagrange=1.)

        # store groups as a class variable since the non-group lasso doesn't
        self.groups = groups

        self._initial_omega = perturb

        # gaussian randomization
        self.randomizer = randomizer

        # Whether a Jacobian is needed for gaussian_query
        # this should always be true for group Lasso
        self.useJacobian = useJacobian

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        # solve the randomized version of group lasso
        (self.observed_soln,
         self.observed_subgrad) = self._solve_randomized_problem(perturb=perturb,
                                                                solve_args=solve_args)

        # initialize variables
        active_groups = []  # active group labels
        active_dirs = {}  # dictionary: keys are group labels, values are unit-norm coefficients
        unpenalized = []  # selected groups with no penalty
        overall = np.ones(self.nfeature, np.bool)  # mask of active features
        ordered_groups = []  # active group labels sorted by label
        ordered_opt = []  # gamma's ordered by group labels
        ordered_vars = []  # indices "ordered" by sorting group labels

        tol = 1.e-20

        _, self.prec_randomizer = self.randomizer.cov_prec

        # now we are collecting the directions and norms of the active groups
        for g in sorted(np.unique(self.groups)):  # g is group label

            group_mask = self.groups == g
            soln = self.observed_soln  # do not need to keep setting this

            if norm(soln[group_mask]) > tol * norm(soln):  # is group g appreciably nonzero
                ordered_groups.append(g)

                # variables in active group
                ordered_vars.extend(np.flatnonzero(group_mask))

                if self.penalty.weights[g] == 0:
                    unpenalized.append(g)

                else:
                    active_groups.append(g)
                    active_dirs[g] = soln[group_mask] / norm(soln[group_mask])

                ordered_opt.append(norm(soln[group_mask]))
            else:
                overall[group_mask] = False

        self.selection_variable = {'directions': active_dirs,
                                   'active_groups': active_groups}  # kind of redundant with keys of active_dirs

        self._ordered_groups = ordered_groups

        # exception if no groups are selected
        if len(self.selection_variable['active_groups']) == 0:
            return np.sign(soln), soln

        # otherwise continue as before
        self.observed_opt_state = np.hstack(ordered_opt)  # gammas as array
        num_opt_var = self.observed_opt_state.shape[0]

        _beta_unpenalized = restricted_estimator(self.loglike,  # refit OLS on E
                                                 overall,
                                                 solve_args=solve_args)

        beta_bar = np.zeros(self.nfeature)
        beta_bar[overall] = _beta_unpenalized  # refit OLS beta with zeros
        self._beta_full = beta_bar

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))  # all 1's for LS
        opt_linearNoU = np.dot(X.T, X[:, ordered_vars] * W[:, np.newaxis])

        for i, var in enumerate(ordered_vars):
            opt_linearNoU[var, i] += self.ridge_term

        self.observed_score_state = -opt_linearNoU.dot(_beta_unpenalized)
        self.observed_score_state[~overall] += self.loglike.smooth_objective(beta_bar, 'grad')[~overall]

        active_signs = np.sign(self.observed_soln)
        active = np.flatnonzero(active_signs)
        self.active = active

        # compute part of hessian

        _hessian, _hessian_active, _hessian_unpen = _compute_hessian(self.loglike,
                                                                     beta_bar,
                                                                     active,
                                                                     unpenalized)

        def compute_Vg(ug):
            pg = ug.size  # figure out size of g'th group
            if pg > 1:
                Z = np.column_stack((ug, np.eye(pg, pg - 1)))
                Q, _ = qr(Z)
                Vg = Q[:, 1:]  # drop the first column
            else:
                Vg = np.zeros((1, 0))  # if the group is size one, the orthogonal complement is empty
            return Vg

        def compute_Lg(g):
            pg = active_dirs[g].size
            Lg = self.penalty.weights[g] * np.eye(pg)
            return Lg

        sorted_active_dirs = collections.OrderedDict(sorted(active_dirs.items()))

        Vs = [compute_Vg(ug) for ug in sorted_active_dirs.values()]
        V = block_diag(*Vs)  # unpack the list
        Ls = [compute_Lg(g) for g in sorted_active_dirs]
        L = block_diag(*Ls)  # unpack the list
        XE = X[:, ordered_vars]  # changed to ordered_vars
        Q = XE.T.dot(self._W[:, None] * XE)
        QI = inv(Q)
        C = V.T.dot(QI).dot(L).dot(V)

        self.XE = XE
        self.Q = Q
        self.QI = QI
        self.C = C

        U = block_diag(*[ug for ug in sorted_active_dirs.values()]).T

        self.opt_linear = opt_linearNoU.dot(U)
        self.active_dirs = active_dirs
        self.ordered_vars = ordered_vars

        self.linear_part = -np.eye(self.observed_opt_state.shape[0])
        self.offset = np.zeros(self.observed_opt_state.shape[0])


        # For setting up implied Gaussian

        self._unscaled_cov_score = _hessian
        self.num_opt_var = num_opt_var

        self._setup_sampler_data = (self.linear_part,
                                    self.offset,
                                    self.opt_linear,
                                    self.observed_subgrad)

        self.solved = True

        return active_signs, soln

    def setup_inference(self,
                        dispersion):
        # Using gaussian_query's inherited functions
        if self.num_opt_var > 0:
            self._setup_sampler(*self._setup_sampler_data,
                                dispersion=dispersion)

    def _solve_randomized_problem(self,
                                  perturb=None,
                                  solve_args={'tol': 1.e-15, 'min_its': 100}):

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

        # if all groups are size 1, set up lasso penalty and run usual lasso solver... (see existing code)...

        observed_soln = problem.solve(quad, **solve_args)
        observed_subgrad = -(self.loglike.smooth_objective(observed_soln,
                                                          'grad') +
                            quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

    @staticmethod
    def gaussian(X,
                 Y,
                 groups,
                 weights,
                 sigma=1.,
                 quadratic=None,
                 ridge_term=0.,
                 perturb=None,
                 use_lasso=True,  # should lasso solver be used when applicable - defaults to True
                 randomizer_scale=None):

        loglike = rr.glm.gaussian(X, Y, coef=1. / sigma ** 2, quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(Y) * np.sqrt(mean_diag) / np.sqrt(n - 1)

        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(Y) * np.sqrt(n / (n - 1.))

        randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)

        return group_lasso(loglike,
                           groups,
                           weights,
                           ridge_term,
                           randomizer,
                           use_lasso,
                           perturb)

def _check_groups(groups):
    """Make sure that the user-specific groups are ok
    There are a number of assumptions that group_lasso makes about
    how groups are specified. Specifically, we assume that
    `groups` is a 1-d array_like of integers that are sorted in
    increasing order, start at 0, and have no gaps (e.g., if there
    is a group 2 and a group 4, there must also be at least one
    feature in group 3).
    This function checks the user-specified group scheme and
    raises an exception if it finds any problems.
    Sorting feature groups is potentially tedious for the user and
    in future we might do this for them.
    """

    # check array_like
    agroups = np.array(groups)

    # check dimension
    if len(agroups.shape) != 1:
        raise ValueError("Groups are not a 1D array_like")

    # check sorted
    if np.any(agroups[:-1] > agroups[1:]) < 0:
        raise ValueError("Groups are not sorted")

    # check integers
    if not np.issubdtype(agroups.dtype, np.integer):
        raise TypeError("Groups are not integers")

    # check starts with 0
    if not np.amin(agroups) == 0:
        raise ValueError("First group is not 0")

    # check for no skipped groups
    if not np.all(np.diff(np.unique(agroups)) == 1):
        raise ValueError("Some group is skipped")
