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

class group_lasso(object):

    def __init__(self,
                 loglike,
                 groups,
                 weights,
                 ridge_term,
                 randomizer,
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

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):

        # solve the randomized version of group lasso
        (self.initial_soln,
         self.initial_subgrad) = self._solve_randomized_problem(perturb=perturb,
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

        _, self.randomizer_prec = self.randomizer.cov_prec

        # now we are collecting the directions and norms of the active groups
        for g in sorted(np.unique(self.groups)):  # g is group label

            group_mask = self.groups == g
            soln = self.initial_soln  # do not need to keep setting this

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

        _beta_unpenalized = restricted_estimator(self.loglike,  # refit OLS on E
                                                 overall,
                                                 solve_args=solve_args)

        beta_bar = np.zeros(self.nfeature)
        # refit OLS beta with zeros: set active features, while leave nonactive ones as zero
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))  # all 1's for LS
        # YH: X^T X_E, without multiplying u, covariates ordered by groups
        opt_linearNoU = np.dot(X.T, X[:, ordered_vars] * W[:, np.newaxis])

        for i, var in enumerate(ordered_vars):
            opt_linearNoU[var, i] += self.ridge_term

        opt_offset = self.initial_subgrad

        # YH: - X^T X_E Beta_E^LS = N_E - X^T Y
        # NE = -X'(Y-X_E Beta_E^LS) (nonzero only on -E)
        self.observed_score_state = -opt_linearNoU.dot(_beta_unpenalized)
        # YH: - X^T X_E Beta_E^LS + (-N_E)      (checked)
        self.observed_score_state[~overall] += self.loglike.smooth_objective(beta_bar, 'grad')[~overall]
        # YH: # self.observed_score_state should equal -X^T Y in the end

        active_signs = np.sign(self.initial_soln)
        active = np.flatnonzero(active_signs)
        self.active = active

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
        # YH: V, L added for paired group lasso implementations
        self.V = V
        self.L = L

        # YH: Each block in U has only one column
        #     ug's are sorted by groups
        U = block_diag(*[ug for ug in sorted_active_dirs.values()]).T

        # YH: U added for paired group lasso implementations
        self.U = U

        # YH: X^T sum(X_g u_g) = B in the paper
        self.opt_linear = opt_linearNoU.dot(U)  #self.opt_linear.dot(self.observed_opt_state) = X^T sum(X gamma u)
        self.active_dirs = active_dirs
        self.opt_offset = opt_offset
        self.ordered_vars = ordered_vars
        self.overall = overall

        # YH: observed_opt_state: gammas as array
        #     linear_part: negative identity with the same dimension of #groups
        #     offset: 0 vector of #groups x 1
        self.linear_part = -np.eye(self.observed_opt_state.shape[0])
        self.offset = np.zeros(self.observed_opt_state.shape[0])

        return active_signs, soln

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

        initial_soln = problem.solve(quad, **solve_args)
        initial_subgrad = -(self.loglike.smooth_objective(initial_soln,
                                                          'grad') +
                            quad.objective(initial_soln, 'grad'))

        return initial_soln, initial_subgrad

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

    def _setup_implied_gaussian(self):

        # YH: Precision matrix of randomizer
        _, prec = self.randomizer.cov_prec

        # YH: opt_linear = X^T sum(X_g u_g) = B in the paper
        # YH: cond_precision is the quadratic form (B^T Omega^-1 B) = Omega_bar^-1
        # YH: cond_cov is Omega_bar
        # YH: logdens_linear = Omega_bar B^T Omega^-1 = -b_bar without c

        # YH: if the precision "matrix" is a scalar
        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = self.opt_linear.T.dot(self.opt_linear) * prec
            cond_cov = inv(cond_precision)
            logdens_linear = cond_cov.dot(self.opt_linear.T) * prec
        # YH: if there is a nontrivial covariance structure for the randomization
        else:
            cond_precision = self.opt_linear.T.dot(prec.dot(self.opt_linear))
            cond_cov = inv(cond_precision)
            logdens_linear = cond_cov.dot(self.opt_linear.T).dot(prec)
            # YH: Omega_bar B^T Omega^-1 = -b_bar without c

        # YH: observed_score_state is -X^T Y, opt_offset is the subgrad
        # YH: self.observed_score_state + self.opt_offset = -X^T Y + subgrad
        # cond_mean = -Omega_bar B^T Omega^-1 (A beta_E^(LS) + b_bar)
        #           = -Omega_bar B^T Omega^-1 (-X^T Y + subgrad)
        #           = A_bar beta_E^(LS) + b_bar
        # self.observed_score_state should equal -X^T Y
        cond_mean = -logdens_linear.dot(self.observed_score_state + self.opt_offset)
        self.cond_mean = cond_mean                  # A_bar beta_E^(LS) + b_bar
        self.cond_cov = cond_cov                    # Omega_bar
        self.cond_precision = cond_precision        # Omega_bar^-1
        self.logdens_linear = logdens_linear        # Omega_bar B^T Omega^-1

        return cond_mean, cond_cov, cond_precision, logdens_linear

    def selective_MLE(self,
                      solve_args={'tol': 1.e-12},
                      level=0.9,
                      useJacobian=True,
                      dispersion=None):

        """Do selective_MLE for group_lasso
        Note: this masks the selective_MLE inherited from query
        because that is not adapted for the group_lasso. Also, assumes
        you have already run the fit method since this uses results
        from that method.
        Parameters
        ----------
        observed_target: from selected_targets
        target_cov: from selected_targets
        target_cov_score: from selected_targets
        init_soln:  (opt_state) initial (observed) value of optimization variables
        cond_mean: conditional mean of optimization variables (model on _setup_implied_gaussian)
        cond_cov: conditional variance of optimization variables (model on _setup_implied_gaussian)
        logdens_linear: (model on _setup_implied_gaussian)
        linear_part: like A_scaling (from lasso)
        offset: like b_scaling (from lasso)
        solve_args: passed on to solver
        level: level of confidence intervals
        useC: whether to use python or C solver
        JacobianPieces: (use self.C defined in fitting)
        """

        self._setup_implied_gaussian()  # Calculate useful quantities

        # YH: dispersion is the variance (omega^2) of the presumably true OLS model
        # YH: observed_target is the OLS solution on E
        # YH: target_cov = self.QI * dispersion
        # YH: crosscov_target_score = - (X^T XE (XE^T XE)^{-1})^T * dispersion
        #                           = - (XE^T XE)^{-1} XE^T X * dispersion
        # YH: alternatives = ['twosided'] * len(self.active)
        (observed_target, target_cov, target_score_cov, alternatives) = self.selected_targets(dispersion)

        init_soln = self.observed_opt_state  # just the gammas
        # YH: cond_mean =  A_bar beta_E^(LS) + b_bar
        cond_mean = self.cond_mean
        # YH: cond_cov is Omega_bar
        cond_cov = self.cond_cov
        # YH: logdens_linear = Omega_bar B^T Omega^-1 = -b_bar without c
        logdens_linear = self.logdens_linear
        # YH: self.linear_part should be A_bar
        linear_part = self.linear_part
        # YH: self.offset should be b_bar
        offset = self.offset

        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(observed_target)
        prec_target = inv(target_cov)

        # YH: prec_opt is omega_bar^{-1} (omega_bar is a covariance)
        prec_opt = self.cond_precision

        # YH: -X^T Y + subgrad
        score_offset = self.observed_score_state + self.opt_offset

        # target_lin determines how the conditional mean of optimization variables
        # vary with target
        # logdens_linear determines how the argument of the optimization density
        # depends on the score, not how the mean depends on score, hence the minus sign

        # YH: target_score_cov.T = - (X^T XE (XE^T XE)^{-1}) * dispersion
        # YH: so target_linear = - (X^T XE (XE^T XE)^{-1}) (XE^T XE)
        #                      = - X^T XE = A
        target_linear = target_score_cov.T.dot(prec_target)
        # YH: target_offset = -X^T Y + subgrad + X^T XE (XE^T XE)^{-1} XE^T
        # YH: target_offset = c
        target_offset = score_offset - target_linear.dot(observed_target)

        # YH: A_bar
        target_lin = - logdens_linear.dot(target_linear)
        # YH: b_bar
        target_off = cond_mean - target_lin.dot(observed_target)

        if np.asarray(self.randomizer_prec).shape in [(), (0,)]:
            # YH: A^T  Omega^-1 c
            _P = target_linear.T.dot(target_offset) * self.randomizer_prec
            # YH: theta_bar^{-1}
            _prec = prec_target + (target_linear.T.dot(target_linear) * self.randomizer_prec) - target_lin.T.dot(
                prec_opt).dot(
                target_lin)
        else:
            # YH: A^T  Omega^-1 c
            _P = target_linear.T.dot(self.randomizer_prec).dot(target_offset)
            # YH: theta_bar^{-1}
            _prec = prec_target + (target_linear.T.dot(self.randomizer_prec).dot(target_linear)) - target_lin.T.dot(
                prec_opt).dot(target_lin)

        # YH: -s_bar, replacing theta_bar with Sigma
        C = target_cov.dot(_P - target_lin.T.dot(prec_opt).dot(target_off))

        # YH: prec_opt is omega_bar^{-1}
        conjugate_arg = prec_opt.dot(cond_mean)

        val, soln, hess = solve_barrier_affine_jacobian_py(conjugate_arg,
                                                           prec_opt,
                                                           init_soln,
                                                           linear_part,
                                                           offset,
                                                           self.C,
                                                           self.active_dirs,
                                                           useJacobian,
                                                           **solve_args)

        final_estimator = target_cov.dot(_prec).dot(observed_target) \
                          + target_cov.dot(target_lin.T.dot(prec_opt.dot(cond_mean - soln))) + C

        unbiased_estimator = target_cov.dot(_prec).dot(observed_target) + target_cov.dot(
            _P - target_lin.T.dot(prec_opt).dot(target_off))

        L = target_lin.T.dot(prec_opt)
        observed_info_natural = _prec + L.dot(target_lin) - L.dot(hess.dot(L.T))

        observed_info_mean = target_cov.dot(observed_info_natural.dot(target_cov))

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

        pvalues = ndist.cdf(Z_scores)

        pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

        alpha = 1 - level
        quantile = ndist.ppf(1 - alpha / 2.)

        intervals = np.vstack([final_estimator -
                               quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator +
                               quantile * np.sqrt(np.diag(observed_info_mean))]).T

        log_ref = val + conjugate_arg.T.dot(cond_cov).dot(conjugate_arg) / 2.

        result = pd.DataFrame({'MLE': final_estimator,
                               'SE': np.sqrt(np.diag(observed_info_mean)),
                               'Zvalue': Z_scores,
                               'pvalue': pvalues,
                               'lower_confidence': intervals[:, 0],
                               'upper_confidence': intervals[:, 1],
                               'unbiased': unbiased_estimator})

        return result, observed_info_mean, log_ref

    def selected_targets(self,
                         dispersion=None,
                         solve_args={'tol': 1.e-12, 'min_its': 50}):

        X, y = self.loglike.data
        n, p = X.shape

        XE = self.XE
        Q = self.Q          # YH: Q = XE^T XE
        # YH: ordered_vars: variable indices ordered by group memberships
        # YH: observed_target is the OLS solution on E
        observed_target = restricted_estimator(self.loglike, self.ordered_vars, solve_args=solve_args)
        _score_linear = -XE.T.dot(self._W[:, None] * X).T  # YH: -X^T XE, W is 1 for linear models
        alternatives = ['twosided'] * len(self.active)

        # YH: dispersion is the variance (omega^2) of the presumably true OLS model
        if dispersion is None:  # use Pearson's X^2
            dispersion = ((y - self.loglike.saturated_loss.mean_function(
                XE.dot(observed_target))) ** 2 / self._W).sum() / (n - XE.shape[1])

        cov_target = self.QI * dispersion
        # YH: crosscov_target_score = - (X^T XE (XE^T XE)^{-1})^T * dispersion
        #                           = - (XE^T XE)^{-1} XE^T X  * dispersion
        crosscov_target_score = _score_linear.dot(self.QI).T * dispersion

        return (observed_target,
                cov_target,
                crosscov_target_score,
                alternatives)


class approximate_grid_inference(object):

    def __init__(self,
                 query,
                 dispersion,
                 solve_args={'tol': 1.e-12},
                 useIP=True):

        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        query : `gaussian_query`
            A Gaussian query which has information
            to describe implied Gaussian.
        observed_target : ndarray
            Observed estimate of target.
        target_cov : ndarray
            Estimated covaraince of target.
        target_score_cov : ndarray
            Estimated covariance of target and score of randomized query.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        self.solve_args = solve_args

        result, inverse_info = query.selective_MLE(dispersion=dispersion)[:2]

        # YH: should be A_bar
        self.linear_part = query.linear_part
        # YH: should be b_bar
        self.offset = query.offset

        # YH: logdens_linear = Omega_bar B^T Omega^-1 = -b_bar without c
        self.logdens_linear = query.logdens_linear
        # YH: cond_mean = A_bar beta_E^(LS) + b_bar
        self.cond_mean = query.cond_mean
        # YH: prec_opt is omega_bar^{-1}
        self.prec_opt = np.linalg.inv(query.cond_cov)
        # YH: cond_cov is Omega_bar
        self.cond_cov = query.cond_cov

        # YH: V.T.dot(QI).dot(L).dot(V), V: U_bar, L: Lambda, QI: Q^-1
        self.C = query.C
        # dictionary: keys are group labels, values are unit-norm coefficients
        self.active_dirs = query.active_dirs

        # YH: dispersion is the variance (omega^2) of the presumably true OLS model
        # YH: observed_target is the OLS solution on E
        # YH: cov_target = self.QI * dispersion
        # YH: crosscov_target_score = - (X^T XE (XE^T XE)^{-1})^T * dispersion
        #                           = - (XE^T XE)^{-1} XE^T X * dispersion
        # YH: alternatives = ['twosided'] * len(self.active)
        (observed_target, target_cov, target_score_cov, alternatives) = query.selected_targets(dispersion)
        self.observed_target = observed_target
        self.target_score_cov = target_score_cov
        self.target_cov = target_cov

        self.init_soln = query.observed_opt_state

        self.randomizer_prec = query.randomizer_prec
        # YH: observed_score_state = -N_E, opt_offset = subgrad
        # YH: so score_offset = c in the paper
        self.score_offset = query.observed_score_state + query.opt_offset

        # ntarget = |E|
        self.ntarget = ntarget = target_cov.shape[0]
        # inverse_info is the inverse information matrix in solving the Barrier problem
        # _scale determines the range of the grids
        _scale = 4 * np.sqrt(np.diag(inverse_info))

        if useIP == False:
            ngrid = 1000
            self.stat_grid = np.zeros((ntarget, ngrid))
            for j in range(ntarget):
                # Return evenly spaced numbers over a specified interval
                # self.stat_grid[j, :] is a 1000 x 1 np array of numbers
                self.stat_grid[j, :] = np.linspace(observed_target[j] - 1.5 * _scale[j],
                                                   observed_target[j] + 1.5 * _scale[j],
                                                   num=ngrid)
        else:
            ngrid = 100
            self.stat_grid = np.zeros((ntarget, ngrid))
            for j in range(ntarget):
                self.stat_grid[j, :] = np.linspace(observed_target[j] - 1.5 * _scale[j],
                                                   observed_target[j] + 1.5 * _scale[j],
                                                   num=ngrid)

        self.opt_linear = query.opt_linear
        self.useIP = useIP

    def summary(self,
                alternatives=None,
                parameter=None,
                level=0.9):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        """

        if parameter is not None:
            pivots = self._approx_pivots(parameter,
                                        alternatives=alternatives)
        else:
            pivots = None

        pvalues = self._approx_pivots(np.zeros_like(self.observed_target),
                                      alternatives=alternatives)
        lower, upper = self._approx_intervals(level=level)

        result = pd.DataFrame({'target': self.observed_target,
                               'pvalue': pvalues,
                               'lower_confidence': lower,
                               'upper_confidence': upper})

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

    def log_reference(self,
                      observed_target,
                      target_cov,
                      target_score_cov,
                      grid):

        """
        Approximate the log of the reference density on a grid.
        """

        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        # YH: prec_target = (XE^T XE) / dispersion
        prec_target = np.linalg.inv(target_cov)
        # YH: A_bar
        target_lin = - self.logdens_linear.dot(target_score_cov.T.dot(prec_target))

        ref_hat = []

        # Iterate over dimensions of target
        for k in range(grid.shape[0]):
            # in the usual D = N + Gamma theta.hat,
            # target_lin is "something" times Gamma,
            # where "something" comes from implied Gaussian
            # cond_mean is "something" times D
            # Gamma is target_score_cov.T.dot(prec_target)

            num_opt = self.prec_opt.shape[0]
            num_con = self.linear_part.shape[0]

            # YH: A_bar @ (.) + A_bar beta_E^(LS) + b_bar
            cond_mean_grid = (target_lin.dot(np.atleast_1d(grid[k] - observed_target)) +
                              self.cond_mean)

            #direction for decomposing o
            # YH: prec_opt is omega_bar^{-1}, logdens_linear = Omega_bar B^T Omega^-1,
            #     target_score_cov = - (XE^T XE)^{-1} XE^T X * dispersion
            #     eta: - B^T Omega^-1 A QI * dispersion
            eta = -self.prec_opt.dot(self.logdens_linear.dot(target_score_cov.T))

            implied_mean = np.asscalar(eta.T.dot(cond_mean_grid))
            implied_cov = np.asscalar(eta.T.dot(self.cond_cov).dot(eta))
            implied_prec = 1./implied_cov

            _A = self.cond_cov.dot(eta) * implied_prec
            R = np.identity(num_opt) - _A.dot(eta.T)

            A = self.linear_part.dot(_A).reshape((-1,))
            b = self.offset-self.linear_part.dot(R).dot(self.init_soln)

            conjugate_arg = implied_mean * implied_prec

            # YH: solver: solve_barrier_affine_py
            # YH: Approximating selection probability using (4)/(OA) in MCMC-free (?)
            val, soln, _ = solver(np.asarray([conjugate_arg]),
                                  np.reshape(implied_prec, (1,1)),
                                  eta.T.dot(self.init_soln),
                                  A.reshape((A.shape[0],1)),
                                  b,
                                  **self.solve_args)

            gamma_ = _A.dot(soln) + R.dot(self.init_soln)
            log_jacob = jacobian_grad_hess(gamma_, self.C, self.active_dirs)

            ref_hat.append(-val - ((conjugate_arg ** 2) * implied_cov)/ 2. + log_jacob[0])

        return np.asarray(ref_hat)

    def _construct_families(self):

        self._construct_density()

        self._families = []

        for m in range(self.ntarget):
            p = self.target_score_cov.shape[1]
            observed_target_uni = (self.observed_target[m]).reshape((1,))

            target_cov_uni = (np.diag(self.target_cov)[m]).reshape((1, 1))
            target_score_cov_uni = self.target_score_cov[m, :].reshape((1, p))

            var_target = 1. / ((self.precs[m])[0, 0])

            # Evaluate h(t) over the grids
            log_ref = self.log_reference(observed_target_uni,
                                         target_cov_uni,
                                         target_score_cov_uni,
                                         self.stat_grid[m])
            if self.useIP == False:
                logW = (log_ref - 0.5 * (self.stat_grid[m] - self.observed_target[m]) ** 2 / var_target)
                logW -= logW.max()
                self._families.append(discrete_family(self.stat_grid[m],
                                                      np.exp(logW)))
            else:
                # YH: Quadratic interpolation of h(t)
                approx_fn = interp1d(self.stat_grid[m],
                                     log_ref,
                                     kind='quadratic',
                                     bounds_error=False,
                                     fill_value='extrapolate')

                grid = np.linspace(self.stat_grid[m].min(), self.stat_grid[m].max(), 1000)
                # YH: conditional marginal density: -1/2 * (t-b)^2 / sigma_j^2 + h(t),
                # YH: (?) replacing b (parameter) with the observed quantity
                logW = (approx_fn(grid) -
                        0.5 * (grid - self.observed_target[m]) ** 2 / var_target)

                logW -= logW.max()

                # YH: This is a discrete approximation of a family
                #     with sufficient statistic being just x and
                #     weights being w, and theta default to 0.
                #     That is, a discrete pmf to approximate the pdf given by logW
                self._families.append(discrete_family(grid,
                                                      np.exp(logW)))

    def _approx_pivots(self,
                       mean_parameter,
                       alternatives=None):

        if not hasattr(self, "_families"):
            self._construct_families()

        if alternatives is None:
            alternatives = ['twosided'] * self.ntarget

        pivot = []

        for m in range(self.ntarget):

            family = self._families[m]
            var_target = 1. / ((self.precs[m])[0, 0])

            mean = self.S[m].dot(mean_parameter[m].reshape((1,))) + self.r[m]

            _cdf = family.cdf((mean[0] - self.observed_target[m]) / var_target, x=self.observed_target[m])
            print("variable completed ", m)

            if alternatives[m] == 'twosided':
                pivot.append(2 * min(_cdf, 1 - _cdf))
            elif alternatives[m] == 'greater':
                pivot.append(1 - _cdf)
            elif alternatives[m] == 'less':
                pivot.append(_cdf)
            else:
                raise ValueError('alternative should be in ["twosided", "less", "greater"]')
        return pivot

    def _approx_intervals(self,
                   level=0.9):

        if not hasattr(self, "_families"):
            self._construct_families()

        lower, upper = [], []

        for m in range(self.ntarget):
            # construction of intervals from families follows `selectinf.learning.core`
            family = self._families[m]
            observed_target = self.observed_target[m]

            l, u = family.equal_tailed_interval(observed_target,
                                                alpha=1 - level)

            var_target = 1. / ((self.precs[m])[0, 0])

            lower.append(l * var_target + observed_target)
            upper.append(u * var_target + observed_target)

        return np.asarray(lower), np.asarray(upper)

    ### Private method
    ## YH: WHAT DOES THIS FUNCTIO DO?
    def _construct_density(self):

        precs = {}
        S = {}
        r = {}

        # YH: target_score_cov = - (XE^T XE)^{-1} XE^T X * dispersion
        p = self.target_score_cov.shape[1]

        # Loop over selected targets (betas)
        for m in range(self.ntarget):
            # YH: observed_target is the OLS solution on E
            observed_target_uni = (self.observed_target[m]).reshape((1,))
            # YH: target_cov = self.QI * dispersion
            #     This is the variance of the m-th target
            target_cov_uni = (np.diag(self.target_cov)[m]).reshape((1, 1))
            prec_target = 1. / target_cov_uni
            target_score_cov_uni = self.target_score_cov[m, :].reshape((1, p))

            # YH: - dispersion/target_cov_m * (XE^T XE)^{-1}_(m,:) XE^T X
            target_linear = target_score_cov_uni.T.dot(prec_target)
            # YH: score_offset = c in the paper
            #     target_offset = c - dispersion*beta_m/target_cov_m * (XE^T XE)^{-1}_(m,:) XE^T X
            target_offset = (self.score_offset - target_linear.dot(observed_target_uni)).reshape(
                (target_linear.shape[0],))

            # YH: logdens_linear = Omega_bar B^T Omega^-1
            #     cond_mean = A_bar beta_E^(LS) + b_bar
            #     target_lin = dispersion/target_cov_m * Omega_bar B^T Omega^-1 (XE^T XE)^{-1}_(m,:) XE^T X
            #     target_off = A_bar beta_E^(LS) + b_bar - target_lin * beta_m
            target_lin = -self.logdens_linear.dot(target_linear)
            target_off = (self.cond_mean - target_lin.dot(observed_target_uni)).reshape((target_lin.shape[0],))

            # YH:
            _prec = prec_target + (target_linear.T.dot(target_linear) * self.randomizer_prec) - target_lin.T.dot(
                self.prec_opt).dot(target_lin)

            _P = target_linear.T.dot(target_offset) * self.randomizer_prec
            _r = (1. / _prec).dot(target_lin.T.dot(self.prec_opt).dot(target_off) - _P)
            _S = np.linalg.inv(_prec).dot(prec_target)

            S[m] = _S
            r[m] = _r
            precs[m] = _prec

        self.precs = precs
        self.S = S
        self.r = r


def solve_barrier_affine_jacobian_py(conjugate_arg,
                                     precision,
                                     feasible_point,
                                     con_linear,
                                     con_offset,
                                     C,
                                     active_dirs,
                                     useJacobian=True,
                                     step=1,
                                     nstep=2000,
                                     min_its=500,
                                     tol=1.e-12):
    """
    This needs to be updated to actually use the Jacobian information (in self.C)
    arguments
    conjugate_arg: \\bar{\\Sigma}^{-1} \bar{\\mu}
    precision:  \\bar{\\Sigma}^{-1}
    feasible_point: gamma's from fitting
    con_linear: linear part of affine constraint used for barrier function
    con_offset: offset part of affine constraint used for barrier function
    C: V^T Q^{-1} \\Lambda V
    active_dirs:
    """
    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    def objective(gs):
        p1 = -gs.T.dot(conjugate_arg)
        p2 = gs.T.dot(precision).dot(gs) / 2.
        if useJacobian:
            p3 = - jacobian_grad_hess(gs, C, active_dirs)[0]
        else:
            p3 = 0
        p4 = log(1. + 1. / ((con_offset - con_linear.dot(gs)) / scaling)).sum()
        return p1 + p2 + p3 + p4

    def grad(gs):
        p1 = -conjugate_arg + precision.dot(gs)
        p2 = -con_linear.T.dot(1. / (scaling + con_offset - con_linear.dot(gs)))
        if useJacobian:
            p3 = - jacobian_grad_hess(gs, C, active_dirs)[1]
        else:
            p3 = 0
        p4 = 1. / (con_offset - con_linear.dot(gs))
        return p1 + p2 + p3 + p4

    def barrier_hessian(gs):  # contribution of barrier and jacobian to hessian
        p1 = con_linear.T.dot(np.diag(-1. / ((scaling + con_offset - con_linear.dot(gs)) ** 2.)
                                      + 1. / ((con_offset - con_linear.dot(gs)) ** 2.))).dot(con_linear)
        if useJacobian:
            p2 = - jacobian_grad_hess(gs, C, active_dirs)[2]
        else:
            p2 = 0
        return p1 + p2

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset - con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')
        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = inv(precision + barrier_hessian(current))
    return current_value, current, hess


# Jacobian calculations
def calc_GammaMinus(gamma, active_dirs):
    """Calculate Gamma^minus (as a function of gamma vector, active directions)
    """
    to_diag = [[g] * (ug.size - 1) for (g, ug) in zip(gamma, active_dirs.values())]
    return block_diag(*[i for gp in to_diag for i in gp])


def jacobian_grad_hess(gamma, C, active_dirs):
    """ Calculate the log-Jacobian (scalar), gradient (gamma.size vector) and hessian (gamma.size square matrix)
    """
    if C.shape == (0, 0):  # when all groups are size one, C will be an empty array
        return 0, 0, 0
    else:
        GammaMinus = calc_GammaMinus(gamma, active_dirs)

        # eigendecomposition
        #evalues, evectors = eig(GammaMinus + C)

        # log Jacobian
        #J = log(evalues).sum()
        J = np.log(np.linalg.det(GammaMinus + C))

        # inverse
        #GpC_inv = evectors.dot(np.diag(1 / evalues).dot(evectors.T))
        GpC_inv = np.linalg.inv(GammaMinus + C)

        # summing matrix (gamma.size by C.shape[0])
        S = block_diag(*[np.ones((1, ug.size - 1)) for ug in active_dirs.values()])

        # gradient
        grad_J = S.dot(GpC_inv.diagonal())

        # hessian
        hess_J = -S.dot(np.multiply(GpC_inv, GpC_inv.T).dot(S.T))

        return J, grad_J, hess_J

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
