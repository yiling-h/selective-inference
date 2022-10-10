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
from .query import gaussian_query
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

        if self.randomizer is not None:
            _, self.prec_randomizer = self.randomizer.cov_prec


        soln = self.observed_soln  # used in the following loop

        # now we are collecting the directions and norms of the active groups
        for g in sorted(np.unique(self.groups)):  # g is group label

            group_mask = self.groups == g

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
        self.active_signs = active_signs
        #self.active_signs[unpenalized] = np.nan
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
        self.U = U
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
                 useJacobian=True,
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
                           useJacobian=useJacobian,
                           use_lasso=use_lasso,
                           perturb=perturb)

    @staticmethod
    def logistic(X,
                 successes,
                 groups,
                 weights,
                 trials=None,
                 quadratic=None,
                 ridge_term=0.,
                 perturb=None,
                 useJacobian=True,
                 randomizer_scale=None,
                 cov_rand=None,
                 use_lasso=True):  # should lasso solver be used when applicable - defaults to True

        loglike = rr.glm.logistic(X,
                                  successes,
                                  trials=trials,
                                  quadratic=quadratic)
        n, p = X.shape

        mean_diag = np.mean((X ** 2).sum(0))
        if ridge_term is None:
            ridge_term = np.std(successes) * np.sqrt(mean_diag) / np.sqrt(n - 1)
        if randomizer_scale is None:
            randomizer_scale = np.sqrt(mean_diag) * 0.5 * np.std(successes) * np.sqrt(n / (n - 1.))

        if cov_rand is None:
            randomizer = randomization.isotropic_gaussian((p,), randomizer_scale)
        else:
            print("Non-isotropic randomization")
            randomizer = randomization.gaussian(cov_rand)

        return group_lasso(loglike,
                           groups,
                           weights,
                           ridge_term=ridge_term,
                           randomizer=randomizer,
                           useJacobian=useJacobian,
                           use_lasso=use_lasso,
                           perturb=perturb)

class split_group_lasso(group_lasso):

    """
    Data split, then group LASSO (i.e. data carving)
    """

    def __init__(self,
                 loglike,
                 groups,
                 weights,
                 proportion_select,
                 randomizer,
                 ridge_term=0,
                 useJacobian=True,
                 use_lasso=True,  # should lasso solver be used where applicable - defaults to True
                 perturb=None,
                 estimate_dispersion=True):

        (self.loglike,
         self.weights,
         self.groups,
         self.proportion_select,
         self.ridge_term) = (loglike,
                             weights,
                             groups,
                             proportion_select,
                             ridge_term)

        self.nfeature = p = self.loglike.shape[0]

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

        self._initial_omega = perturb
        self.randomizer = randomizer

        # Whether a Jacobian is needed for gaussian_query
        # this should always be true for group Lasso
        self.useJacobian = useJacobian

        self.estimate_dispersion = estimate_dispersion

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        signs, soln = group_lasso.fit(self,
                                      solve_args=solve_args,
                                      perturb=perturb)

        # exception if no groups are selected
        if len(self.selection_variable['active_groups']) == 0:
            return signs, soln

        # we then setup up the sampler again
        df_fit = len(self.active)

        if self.estimate_dispersion:
            X, y = self.loglike.data
            n, p = X.shape

            dispersion = 2 * (self.loglike.smooth_objective(self._beta_full,
                                                            'func') /
                              (n - df_fit))

            self.dispersion_ = dispersion
            # run setup again after
            # estimating dispersion

        self.df_fit = df_fit

        return signs, soln

    def setup_inference(self,
                        dispersion):

        if self.df_fit > 0:

            if dispersion is None:
                self._setup_sampler(*self._setup_sampler_data,
                                    dispersion=self.dispersion_)

            else:
                self._setup_sampler(*self._setup_sampler_data,
                                    dispersion=dispersion)

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        # key observation is that the covariance of the added noise is
        # roughly dispersion * (1 - pi) / pi * X^TX (in OLS regression, similar for other
        # models), so the precision is  (X^TX)^{-1} * (pi / ((1 - pi) * dispersion))
        # and prec.dot(opt_linear) = S_E / (dispersion * (1 - pi) / pi)
        # because opt_linear has shape p x E with the columns
        # being those non-zero columns of the solution. Above S_E = np.diag(signs)
        # the conditional precision is S_E Q[E][:,E] * pi / ((1 - pi) * dispersion) S_E
        # and regress_opt is -Q[E][:,E]^{-1} S_E
        # padded with zeros
        # to be E x p

        pi_s = self.proportion_select
        ratio = (1 - pi_s) / pi_s

        ordered_vars = self.ordered_vars

        cond_precision = (opt_linear[ordered_vars]).T.dot(self.U) / (dispersion * ratio)

        assert (np.linalg.norm(cond_precision - cond_precision.T) /
                np.linalg.norm(cond_precision) < 1.e-6)
        cond_cov = np.linalg.inv(cond_precision)
        regress_opt = np.zeros((cond_cov.shape[0],
                                self.nfeature))
        regress_opt[:, ordered_vars] = -cond_cov.dot(self.U.T) / (dispersion * ratio)

        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        ## probably missing a dispersion in the denominator
        prod_score_prec_unnorm = np.identity(self.nfeature) / (dispersion * ratio)

        ## probably missing a multiplicative factor of ratio
        cov_rand = self._unscaled_cov_score * (dispersion * ratio)

        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

        # would be nice to not store these?

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                M1,
                M2,
                M3)

    def _solve_randomized_problem(self,
                                  perturb=None,
                                  solve_args={'tol': 1.e-15, 'min_its': 100}):

        # take a new perturbation if none supplied
        if perturb is not None:
            self._selection_idx = perturb
        if not hasattr(self, "_selection_idx"):
            X, y = self.loglike.data
            total_size = n = X.shape[0]
            pi_s = self.proportion_select
            self._selection_idx = np.zeros(n, np.bool)
            self._selection_idx[:int(pi_s * n)] = True
            np.random.shuffle(self._selection_idx)

        inv_frac = 1 / self.proportion_select
        quad = rr.identity_quadratic(self.ridge_term,
                                     0,
                                     0,
                                     0)

        randomized_loss = self.loglike.subsample(self._selection_idx)
        randomized_loss.coef *= inv_frac

        problem = rr.simple_problem(randomized_loss, self.penalty)

        # if all groups are size 1, set up lasso penalty and run usual lasso solver... (see existing code)...

        observed_soln = problem.solve(quad, **solve_args)
        observed_subgrad = -(randomized_loss.smooth_objective(observed_soln,
                                                          'grad') +
                            quad.objective(observed_soln, 'grad'))

        return observed_soln, observed_subgrad

    @staticmethod
    def gaussian(X,
                 Y,
                 groups,
                 weights,
                 proportion,
                 sigma=1.,
                 quadratic=None,
                 perturb=None,
                 useJacobian=True,
                 use_lasso=True):  # should lasso solver be used when applicable - defaults to True

        loglike = rr.glm.gaussian(X,
                                  Y,
                                  coef=1. / sigma ** 2,
                                  quadratic=quadratic)
        n, p = X.shape

        return split_group_lasso(loglike,
                                 groups,
                                 weights,
                                 proportion_select=proportion,
                                 randomizer=None,
                                 useJacobian=useJacobian,
                                 use_lasso=use_lasso,
                                 perturb=perturb)

    @staticmethod
    def logistic(X,
                 successes,
                 groups,
                 weights,
                 proportion,
                 trials=None,
                 quadratic=None,
                 perturb=None,
                 useJacobian=True,
                 use_lasso=True):  # should lasso solver be used when applicable - defaults to True

        loglike = rr.glm.logistic(X,
                                  successes,
                                  trials=trials,
                                  quadratic=quadratic)
        n, p = X.shape

        return split_group_lasso(loglike,
                                 groups,
                                 weights,
                                 proportion_select=proportion,
                                 randomizer=None,
                                 useJacobian=useJacobian,
                                 use_lasso=use_lasso,
                                 perturb=perturb)

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
