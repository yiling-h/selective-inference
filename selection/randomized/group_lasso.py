from __future__ import print_function
import numpy as np
import regreg.api as rr
from selection.randomized.randomization import randomization
from selection.base import restricted_estimator
from selection.randomized.query import query
from scipy.linalg import block_diag
from numpy import log, sqrt
from numpy.linalg import det, norm, qr, inv, eig
from scipy.stats import norm as ndist


class group_lasso(object):

    def __init__(self,
                 loglike,
                 groups,
                 weights,
                 ridge_term,
                 randomizer,
                 perturb=None):

        _check_groups(groups)   # make sure groups looks sensible

        # log likleihood : quadratic loss
        self.loglike = loglike
        self.nfeature = self.loglike.shape[0]

        # ridge parameter
        self.ridge_term = ridge_term

        # group lasso penalty (from regreg)
        self.penalty = rr.group_lasso(groups,
                                      weights=weights,
                                      lagrange=1.)

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
        active = []             # active group labels
        active_dirs = {}        # dictionary: keys are group labels, values are unit-norm coefficients
        unpenalized = []        # selected groups with no penalty
        overall = np.ones(self.nfeature, np.bool)  # mask of active features
        ordered_groups = []     # active group labels sorted by label
        ordered_opt = []        # gamma's ordered by group labels
        ordered_vars = []       # indices "ordered" by sorting group labels

        tol = 1.e-6

        # now we are collecting the directions and norms of the active groups
        for g in sorted(np.unique(self.penalty.groups)):  # g is group label

            group_mask = self.penalty.groups == g
            soln = self.initial_soln  # do not need to keep setting this

            if norm(soln[group_mask]) > tol * norm(soln):  # is group g appreciably nonzero
                ordered_groups.append(g)

                # variables in active group
                ordered_vars.extend(np.flatnonzero(group_mask))

                if self.penalty.weights[g] == 0:
                    unpenalized.append(g)

                else:
                    active.append(g)
                    active_dirs[g] = soln[group_mask] / norm(soln[group_mask])

                ordered_opt.append(norm(soln[group_mask]))
            else:
                overall[group_mask] = False

        self.selection_variable = {'directions': active_dirs,
                                   'active_groups': active}  # kind of redundant with keys of active_dirs

        self._ordered_groups = ordered_groups

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

        print("CHECK K.K.T. MAP", np.allclose(self._initial_omega,
                                              self.observed_score_state + opt_linearNoU.dot(self.initial_soln[ordered_vars])
                                              + opt_offset))
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

        Vs = [compute_Vg(ug) for ug in active_dirs.values()]
        V = block_diag(*Vs)     # unpack the list
        Ls = [compute_Lg(g) for g in active_dirs]
        L = block_diag(*Ls)     # unpack the list
        XE = X[:, active]       # check if this should be ordered_vars
        Q = XE.T.dot(self._W[:, None] * XE)
        QI = inv(Q)
        C = V.T.dot(QI).dot(L).dot(V)

        self.XE = XE
        self.Q = Q
        self.QI = QI
        self.C = C

        U = block_diag(*[ug for ug in active_dirs.values()]).T

        self.opt_linear = opt_linearNoU.dot(U)
        self.active_dirs = active_dirs
        self.opt_offset = opt_offset
        return active_signs

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
                           randomizer)


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

        linear_part = -np.eye(prec_opt.shape[0])
        offset = np.zeros(prec_opt.shape[0])

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 init_soln,
                                 linear_part,
                                 offset,
                                 self.C,
                                 self.active_dirs,
                                 **solve_args)

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

        return final_estimator, observed_info_mean, Z_scores, pvalues, intervals, ind_unbiased_estimator


    def selected_targets(self,
                         dispersion=None,
                         solve_args={'tol': 1.e-12, 'min_its': 50}):

        X, y = self.loglike.data
        n, p = X.shape

        XE = self.XE
        Q = self.Q
        observed_target = restricted_estimator(self.loglike, self.active, solve_args=solve_args)
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

from selection.tests.instance import gaussian_instance


def gaussian_group_instance(n=100, p=200, sgroup=7, sigma=5, rho=0., signal=7,
                            random_signs=False, df=np.inf,
                            scale=True, center=True,
                            groups=np.arange(20).repeat(10),
                            equicorrelated=True):
    """A testing instance for the group LASSO.


    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\\lambda$ of around 13.5
    corresponds to the theoretical $E(\\|X^T\\epsilon\\|_{\\infty})$
    with $\\epsilon \\sim N(0, \\sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    sgroup : int
        True sparsity (number of active groups)

    groups : array_like (1d, size == p)
        Assignment of features to (non-overlapping) groups

    sigma : float
        Noise level

    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(p).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    equicorrelated: bool
        If true, design in equi-correlated,
        Else design is AR.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.
    """
    from selection.tests.instance import _design
    X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

    if center:
        X -= X.mean(0)[None, :]

    beta = np.zeros(p)
    signal = np.atleast_1d(signal)

    group_labels = np.unique(groups)
    group_active = np.random.choice(group_labels, sgroup, replace=False)

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active)
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if scale:
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)
        sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma, sigmaX


def test_group_lasso(n=200,
                     p=50,
                     signal_fac=3,
                     sgroup=1,
                     groups=np.arange(5).repeat(10),
                     sigma=3,
                     target='selected',
                     rho=0.4,
                     randomizer_scale=1.):

    inst = gaussian_group_instance
    signal = np.sqrt(signal_fac * log(p))

    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      sgroup=sgroup,
                      groups=groups,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)

    weights = dict([(i, sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    conv = group_lasso.gaussian(X,
                                Y,
                                groups,
                                weights,
                                randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()          # fit doesn't actually return anything
    nonzero = conv.selection_variable['directions'].keys()
    print("check ", nonzero)


def solve_barrier_affine_jacobian_py(conjugate_arg,
                                     precision,
                                     feasible_point,
                                     con_linear,
                                     con_offset,
                                     C,
                                     active_dirs,
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
        p2 = gs.T.dot(precision).dot(gs)/2.
        p3 = jacobian_grad_hess(gs, C, active_dirs)[0]
        p4 = log(1. + 1./((con_offset - con_linear.dot(gs)) / scaling)).sum()
        return p1 + p2 + p3 + p4

    def grad(gs):
        p1 = -conjugate_arg + precision.dot(gs)
        p2 = -con_linear.T.dot(1./(scaling + con_offset - con_linear.dot(gs)))
        p3 = jacobian_grad_hess(gs, C, active_dirs)[1]
        p4 = 1./(con_offset - con_linear.dot(gs))
        return p1 + p2 + p3 + p4

    def barrier_hessian(gs):
        p1 = con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(gs))**2.)
                                                 + 1./((con_offset-con_linear.dot(gs))**2.))).dot(con_linear)
        p2 = jacobian_grad_hess(gs, C, active_dirs)[2]
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


test_group_lasso()


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
        # return all the objects
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
    if not np.issubdtype(agroups, np.integer):
        raise TypeError("Groups are not integers")

    # check starts with 0
    if not np.amin(agroups) == 0:
        raise ValueError("First group is not 0")

    # check for no skipped groups
    if not np.diff(np.unique(agroups)) == 1:
        raise ValueError("Some group is skipped")
