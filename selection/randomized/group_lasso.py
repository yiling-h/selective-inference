from __future__ import print_function
import numpy as np
import regreg.api as rr
from selection.randomized.randomization import randomization
from selection.base import restricted_estimator
from scipy.linalg import block_diag
from numpy import log
from numpy.linalg import norm, qr, inv, eig
from scipy.stats import norm as ndist
import collections
from selection.randomized.query import naive_confidence_intervals, naive_pvalues
from scipy.linalg import fractional_matrix_power
import sys

from selectinf.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C


class group_lasso(object):

    def __init__(self,
                 loglike,
                 groups,
                 weights,
                 ridge_term,
                 randomizer,
                 use_lasso=True, # should lasso solver be used where applicable - defaults to True
                 perturb=None):

        _check_groups(groups)   # make sure groups looks sensible

        # log likelihood : quadratic loss
        self.loglike = loglike
        self.nfeature = self.loglike.shape[0]

        # ridge parameter
        self.ridge_term = ridge_term

        # group lasso penalty (from regreg)
        # use regular lasso penalty if all groups are size 1
        if use_lasso and groups.size==np.unique(groups).size:
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
        active_groups = []      # active group labels
        active_dirs = {}        # dictionary: keys are group labels, values are unit-norm coefficients
        unpenalized = []        # selected groups with no penalty
        overall = np.ones(self.nfeature, np.bool)  # mask of active features
        ordered_groups = []     # active group labels sorted by label
        ordered_opt = []        # gamma's ordered by group labels
        ordered_vars = []       # indices "ordered" by sorting group labels

        tol = 1.e-20

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
        if len(self.selection_variable['active_groups'])==0:
            return np.sign(soln), soln
            
        # otherwise continue as before
        self.observed_opt_state = np.hstack(ordered_opt)  # gammas as array

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

        opt_offset = self.initial_subgrad

        self.observed_score_state = -opt_linearNoU.dot(_beta_unpenalized)
        self.observed_score_state[~overall] += self.loglike.smooth_objective(beta_bar, 'grad')[~overall]

        active_signs = np.sign(self.initial_soln)
        active = np.flatnonzero(active_signs)
        self.active = active

        def compute_Vg(ug):
            pg = ug.size    # figure out size of g'th group
            if pg>1:
                Z = np.column_stack((ug, np.eye(pg, pg-1)))
                Q, _ = qr(Z)
                Vg = Q[:, 1:]       # drop the first column
            else:
                Vg = np.zeros((1,0)) # if the group is size one, the orthogonal complement is empty
            return Vg

        def compute_Lg(g):
            pg = active_dirs[g].size
            Lg = self.penalty.weights[g] * np.eye(pg)
            return Lg

        sorted_active_dirs = collections.OrderedDict(sorted(active_dirs.items()))

        Vs = [compute_Vg(ug) for ug in sorted_active_dirs.values()]
        V = block_diag(*Vs)     # unpack the list
        Ls = [compute_Lg(g) for g in sorted_active_dirs]
        L = block_diag(*Ls)     # unpack the list
        XE = X[:, ordered_vars]       # changed to ordered_vars
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
        self.opt_offset = opt_offset
        self.ordered_vars = ordered_vars

        self.linear_part = -np.eye(self.observed_opt_state.shape[0])
        self.offset = np.zeros(self.observed_opt_state.shape[0])

        #print("K.K.T. map", np.allclose(self._initial_omega, self.observed_score_state + self.opt_linear.dot(self.observed_opt_state)
        #                                + self.opt_offset, rtol=1e-03))
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
                 use_lasso=True, # should lasso solver be used when applicable - defaults to True
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

        _, prec = self.randomizer.cov_prec

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = self.opt_linear.T.dot(self.opt_linear) * prec
            cond_cov = inv(cond_precision)
            logdens_linear = cond_cov.dot(self.opt_linear.T) * prec
        else:
            cond_precision = self.opt_linear.T.dot(prec.dot(self.opt_linear))
            cond_cov = inv(cond_precision)
            logdens_linear = cond_cov.dot(self.opt_linear.T).dot(prec)

        cond_mean = -logdens_linear.dot(self.observed_score_state + self.opt_offset)
        self.cond_mean = cond_mean
        self.cond_cov = cond_cov
        self.cond_precision = cond_precision
        self.logdens_linear = logdens_linear

        return cond_mean, cond_cov, cond_precision, logdens_linear

    def selective_MLE(self,
                      solve_args={'tol': 1.e-12},
                      level=0.9,
                      useC=False,
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
        cov_target: from selected_targets
        cov_target_score: from selected_targets
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
        (observed_target, cov_target, cov_target_score, alternatives) = self.selected_targets(dispersion)

        init_soln = self.observed_opt_state  # just the gammas
        cond_mean = self.cond_mean
        cond_cov = self.cond_cov
        logdens_linear = self.logdens_linear
        linear_part = self.linear_part
        offset = self.offset

        if np.asarray(observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(observed_target)
        prec_target = inv(cov_target)

        # target_lin determines how the conditional mean of optimization variables
        # vary with target
        # logdens_linear determines how the argument of the optimization density
        # depends on the score, not how the mean depends on score, hence the minus sign

        target_lin = - logdens_linear.dot(cov_target_score.T.dot(prec_target))
        target_offset = cond_mean - target_lin.dot(observed_target)


        prec_opt = self.cond_precision

        conjugate_arg = prec_opt.dot(cond_mean)

        if useC:
            print("using C")
            solver = solve_barrier_affine_jacobian_C  # not yet implemented
        else:
            print("not using C")
            solver = solve_barrier_affine_jacobian_py

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 init_soln,
                                 linear_part,
                                 offset,
                                 self.C,
                                 self.active_dirs,
                                 useJacobian,
                                 **solve_args)

        log_ref = val + conjugate_arg.T.dot(cond_cov).dot(conjugate_arg) / 2.

        final_estimator = observed_target + cov_target.dot(target_lin.T.dot(prec_opt.dot(cond_mean - soln)))
        ind_unbiased_estimator = observed_target + cov_target.dot(target_lin.T.dot(prec_opt.dot(cond_mean
                                                                                                - init_soln)))
        L = target_lin.T.dot(prec_opt)
        observed_info_natural = prec_target + L.dot(target_lin) - L.dot(hess.dot(L.T))
        observed_info_mean = cov_target.dot(observed_info_natural.dot(cov_target))

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))
        pvalues = ndist.cdf(Z_scores)
        pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

        alpha = 1. - level
        quantile = ndist.ppf(1 - alpha / 2.)
        intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

        return final_estimator, observed_info_mean, Z_scores, pvalues, intervals, ind_unbiased_estimator, log_ref
        
    def naive_inference(self,
                        level=0.9,
                        dispersion=None):
        """This method is analogous to selective_MLE, but performs naive
        inference for group_lasso

        Note: this method assumes you have already run the fit method
        since this uses results from that method.
        """

        self._setup_implied_gaussian()  # Calculate useful quantities
        (observed_target, cov_target, cov_target_score, alternatives) = self.selected_targets(dispersion)

        intervals = naive_confidence_intervals(np.diag(cov_target),
                                               observed_target, level)

        pvalues = naive_pvalues(np.diag(cov_target), observed_target, np.zeros_like(observed_target))

        final_estimator = observed_target
        ind_unbiased_estimator = observed_target
        observed_info_mean = cov_target

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))
        pvalues = ndist.cdf(Z_scores)
        pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

        return final_estimator, observed_info_mean, Z_scores, pvalues, intervals, ind_unbiased_estimator


    def selected_targets(self,
                         dispersion=None,
                         solve_args={'tol': 1.e-12, 'min_its': 50}):

        X, y = self.loglike.data
        n, p = X.shape

        XE = self.XE
        Q = self.Q
        observed_target = restricted_estimator(self.loglike, self.ordered_vars, solve_args=solve_args)
        _score_linear = -XE.T.dot(self._W[:, None] * X).T
        alternatives = ['twosided'] * len(self.active)

        if dispersion is None:  # use Pearson's X^2
            dispersion = ((y - self.loglike.saturated_loss.mean_function(
                XE.dot(observed_target))) ** 2 / self._W).sum() / (n - XE.shape[1])

        cov_target = self.QI * dispersion
        crosscov_target_score = _score_linear.dot(self.QI).T * dispersion

        return (observed_target,
                cov_target,
                crosscov_target_score,
                alternatives)


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
            if np.all(con_offset-con_linear.dot(proposal) > 0):
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
    to_diag = [[g]*(ug.size-1) for (g, ug) in zip(gamma,active_dirs.values())]
    return block_diag(*[i for gp in to_diag for i in gp])

def jacobian_grad_hess(gamma, C, active_dirs):
    """ Calculate the log-Jacobian (scalar), gradient (gamma.size vector) and hessian (gamma.size square matrix)
    """
    if C.shape==(0,0): # when all groups are size one, C will be an empty array
        return 0,0,0
    else:
        GammaMinus = calc_GammaMinus(gamma,active_dirs)

        # eigendecomposition
        evalues,evectors = eig(GammaMinus + C)

        # log Jacobian
        J = log(evalues).sum()

        # inverse
        GpC_inv = evectors.dot(np.diag(1/evalues).dot(evectors.T))

        # summing matrix (gamma.size by C.shape[0])
        S = block_diag(*[np.ones((1,ug.size-1)) for ug in active_dirs.values()])

        # gradient
        grad_J = S.dot(GpC_inv.diagonal())

        # hessian
        hess_J = -S.dot(np.multiply(GpC_inv,GpC_inv.T).dot(S.T))

        return J,grad_J,hess_J


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


class posterior():

    """
    Parameters
    ----------
    observed_target : ndarray
        Observed estimate of target.
    cov_target : ndarray
        Estimated covariance of target.
    cov_target_score : ndarray
        Estimated covariance of target and score of randomized query.
    prior : callable
        A callable object that takes a single argument
        `parameter` of the same shape as `observed_target`
        and returns (gradient of log prior, value of log prior)
    dispersion : float, optional
        A dispersion parameter for likelihood.
    solve_args : dict
        Arguments passed to solver of affine barrier problem.
    """

    def __init__(self,
                 conv,
                 prior,
                 dispersion=1,
                 solve_args={'tol': 1.e-12}):

        self.solve_args = solve_args

        linear_part = conv.linear_part
        offset = conv.offset
        logdens_linear = conv.logdens_linear

        _, self.inverse_info, _, _, _, _, log_ref = conv.selective_MLE(dispersion=dispersion)


        (observed_target, cov_target, cov_target_score, alternatives) = conv.selected_targets(dispersion)

        self.observed_target = observed_target
        self.cov_target_score = cov_target_score
        self.logdens_linear = logdens_linear

        self.ntarget = cov_target.shape[0]
        self.nopt = conv.cond_cov.shape[0]

        self.cond_precision = np.linalg.inv(conv.cond_cov)
        self.prec_target = np.linalg.inv(cov_target)

        self.feasible_point = conv.observed_opt_state
        self.cond_mean = conv.cond_mean
        self.linear_part = linear_part
        self.offset = offset
        self.C = conv.C
        self.active_dirs = conv.active_dirs

        self.initial_estimate = observed_target
        self.dispersion = dispersion
        self.log_ref = log_ref

        self._set_marginal_parameters()

        self.prior = prior

    def log_posterior(self,
                      target_parameter,
                      sigma=1):

        """
        Parameters
        ----------
        target_parameter : ndarray
            Value of parameter at which to evaluate
            posterior and its gradient.
        sigma : ndarray
            Noise standard deviation.
        """

        sigmasq = sigma ** 2
        mean_marginal = self.linear_coef.dot(target_parameter) + self.offset_coef
        prec_marginal = np.linalg.inv(self.cov_marginal)
        conjugate_marginal = prec_marginal.dot(mean_marginal)


        solver = solve_barrier_affine_jacobian_py
        useJacobian = True

        val, soln, hess = solver(conjugate_marginal,
                                 prec_marginal,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 self.C,
                                 self.active_dirs,
                                 useJacobian,
                                 step=1,
                                 nstep=2000,
                                 min_its=500,
                                 tol=1.e-12)

        log_normalizer = -val - mean_marginal.T.dot(prec_marginal).dot(mean_marginal) / 2.

        log_lik = -((self.observed_target - target_parameter).T.dot(self.prec_target).dot(
            self.observed_target - target_parameter)) / 2. \
                  - log_normalizer

        grad_lik = (self.prec_target.dot(self.observed_target) -
                    self.prec_target.dot(target_parameter) \
                    - self.linear_coef.T.dot(prec_marginal.dot(soln) - conjugate_marginal))

        grad_prior, log_prior = self.prior(target_parameter)

        return (self.dispersion * grad_lik / sigmasq + grad_prior,
                self.dispersion * log_lik / sigmasq + log_prior -
                (self.dispersion * self.log_ref / sigmasq))

    def log_posterior_wip(self,
                          target_parameter,
                          sigma=1):

        """
        Parameters
        ----------
        target_parameter : ndarray
            Value of parameter at which to evaluate
            posterior and its gradient.
        sigma : ndarray
            Noise standard deviation.
        """

        sigmasq = sigma ** 2
        mean_marginal = self.linear_coef.dot(target_parameter) + self.offset_coef
        prec_marginal = np.linalg.inv(self.cov_marginal)
        conjugate_marginal = prec_marginal.dot(mean_marginal)


        solver = solve_barrier_affine_C

        val, soln, hess = solver(conjugate_marginal,
                                 prec_marginal,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **self.solve_args)

        log_jacob = jacobian_grad_hess(soln, self.C, self.active_dirs)

        log_normalizer = -val - mean_marginal.T.dot(prec_marginal).dot(mean_marginal) / 2. + log_jacob[0]

        log_lik = -((self.observed_target - target_parameter).T.dot(self.prec_target).dot(
            self.observed_target - target_parameter)) / 2. \
                  - log_normalizer

        grad_lik = (self.prec_target.dot(self.observed_target) -
                    self.prec_target.dot(target_parameter) \
                    - self.linear_coef.T.dot(prec_marginal.dot(soln) - conjugate_marginal)) + \
                   self.linear_coef.T.dot(prec_marginal).dot(log_jacob[1])

        grad_prior, log_prior = self.prior(target_parameter)

        return (self.dispersion * grad_lik / sigmasq + grad_prior,
                self.dispersion * log_lik / sigmasq + log_prior -
                (self.dispersion * self.log_ref / sigmasq))

    ### Private method

    def _set_marginal_parameters(self):
        """
        This works out the implied covariance
        of optimization varibles as a function
        of randomization as well how to compute
        implied mean as a function of the true parameters.
        """
        target_linear = -self.logdens_linear.dot(self.cov_target_score.T.dot(self.prec_target))

        implied_precision = np.zeros((self.ntarget + self.nopt, self.ntarget + self.nopt))
        implied_precision[:self.ntarget, :self.ntarget] = (self.prec_target +
                                                           target_linear.T.dot(self.cond_precision.dot(target_linear)))
        implied_precision[:self.ntarget, self.ntarget:] = -target_linear.T.dot(self.cond_precision)
        implied_precision[self.ntarget:, :self.ntarget] = (-target_linear.T.dot(self.cond_precision)).T
        implied_precision[self.ntarget:, self.ntarget:] = self.cond_precision

        implied_cov = np.linalg.inv(implied_precision)
        self.linear_coef = implied_cov[self.ntarget:, :self.ntarget].dot(self.prec_target)

        target_offset = self.cond_mean - target_linear.dot(self.observed_target)
        M = implied_cov[self.ntarget:, self.ntarget:].dot(self.cond_precision.dot(target_offset))
        N = -target_linear.T.dot(self.cond_precision).dot(target_offset)
        self.offset_coef = implied_cov[self.ntarget:, :self.ntarget].dot(N) + M

        self.cov_marginal = implied_cov[self.ntarget:, self.ntarget:]

    def langevin_sampler(self,
                         nsample=2000,
                         nburnin=100,
                         proposal_scale=None,
                         step=1.):

        state = self.initial_estimate
        stepsize = 1. / (step * self.ntarget)

        if proposal_scale is None:
            proposal_scale = self.inverse_info

        sampler = langevin(state,
                           self.log_posterior_wip,
                           proposal_scale,
                           stepsize,
                           self.dispersion)

        samples = np.zeros((nsample, self.ntarget))

        for i, sample in enumerate(sampler):
            sampler.scaling = self.dispersion
            samples[i, :] = sample.copy()
            sys.stderr.write("sample number " + str(i) + " sample " + str(sample.copy()) + "\n")
            if i == nsample - 1:
                break

        return samples[nburnin:, :]


class langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 proposal_scale,
                 stepsize,
                 scaling):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)
        self.proposal_scale = proposal_scale
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0, scale=1)
        self.sample = np.copy(initial_condition)
        self.scaling = scaling

        self.proposal_sqrt = fractional_matrix_power(self.proposal_scale, 0.5)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):

        while True:

            self.grad_posterior = self.gradient_map(self.state, self.scaling)
            candidate = (self.state + self.stepsize * self.proposal_scale.dot(self.grad_posterior[0])
                        + np.sqrt(2.)* (self.proposal_sqrt.dot(self._noise.rvs(self._shape))) * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate)[0])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                break

        return self.state