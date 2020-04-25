from __future__ import print_function
import numpy as np
import regreg.api as rr
from selection.randomized.randomization import randomization
from selection.base import restricted_estimator
from selection.randomized.query import query


class group_lasso(query):

    def __init__(self,
                 loglike,
                 groups,
                 weights,
                 ridge_term,
                 randomizer,
                 perturb=None):

        # log likleihood : quadratic loss
        self.loglike = loglike
        self.nfeature = p = self.loglike.shape[0]

        # ridge parameter
        self.ridge_term = ridge_term

        # group lasso penalty
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

        active = []
        active_dirs = {}
        unpenalized = []
        overall = np.ones(self.nfeature, np.bool)


        ordered_groups = []
        ordered_opt = []
        ordered_vars = []

        tol = 1.e-6

        # now we are collecting the directions and norms of the active groups
        for g in sorted(np.unique(self.penalty.groups)):

            group = self.penalty.groups == g
            soln = self.initial_soln

            if np.linalg.norm(soln[group]) * tol * np.linalg.norm(soln):

                ordered_groups.append(g)

                # variables in active group
                ordered_vars.extend(np.nonzero(group)[0])

                if self.penalty.weights[g] == 0:
                    unpenalized.append(g)

                else:
                    active.append(g)
                    active_dirs[g] = soln[group] / np.linalg.norm(soln[group])

                ordered_opt.append(np.linalg.norm(soln[group]))
            else:
                overall[group] = False

        self.selection_variable = {'directions': active_dirs,
                                   'active_groups':active}

        self._ordered_groups = ordered_groups

        self.observed_opt_state = np.hstack(ordered_opt)

        _beta_unpenalized = restricted_estimator(self.loglike,
                                                 overall,
                                                 solve_args=solve_args)

        beta_bar = np.zeros(self.nfeature)
        beta_bar[overall] = _beta_unpenalized
        self._beta_full = beta_bar

        X, y = self.loglike.data
        W = self._W = self.loglike.saturated_loss.hessian(X.dot(beta_bar))
        opt_linear = np.dot(X.T, X[:, ordered_vars] * W[:, None])

        for i, var in enumerate(ordered_vars):
            opt_linear[var, i] += self.ridge_term

        opt_offset = self.initial_subgrad

        self.observed_score_state = -opt_linear.dot(_beta_unpenalized)
        self.observed_score_state[~overall] += self.loglike.smooth_objective(beta_bar, 'grad')[~overall]

        print("CHECK K.K.T. MAP", np.allclose(self._initial_omega,
                                           self.observed_score_state + opt_linear.dot(self.initial_soln[ordered_vars])
                                           + opt_offset))
        active_signs = np.sign(self.initial_soln)
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



def selected_targets(loglike,
                     W,
                     active_groups,
                     penalty,
                     sign_info={},
                     dispersion=None,
                     solve_args={'tol': 1.e-12, 'min_its': 50}):

    X, y = loglike.data
    n, p = X.shape
    features = []

    group_assignments = []
    for group in active_groups:
        group_idx = penalty.groups == group
        features.extend(np.nonzero(group_idx)[0])
        group_assignments.extend([group] * group_idx.sum())

    Xfeat = X[:, features]
    Qfeat = Xfeat.T.dot(W[:, None] * Xfeat)
    observed_target = restricted_estimator(loglike, features, solve_args=solve_args)
    cov_target = np.linalg.inv(Qfeat)
    _score_linear = -Xfeat.T.dot(W[:, None] * X).T
    crosscov_target_score = _score_linear.dot(cov_target)
    alternatives = ['twosided'] * len(features)

    if dispersion is None:  # use Pearson's X^2
        dispersion = ((y - loglike.saturated_loss.mean_function(
            Xfeat.dot(observed_target))) ** 2 / W).sum() / (n - Xfeat.shape[1])

    return (observed_target,
            group_assignments,
            cov_target * dispersion,
            crosscov_target_score.T * dispersion,
            alternatives)


def form_targets(target,
                 loglike,
                 W,
                 active_groups,
                 penalty,
                 **kwargs):

    _target = {'full':full_targets,
               'selected':selected_targets,
               'debiased':debiased_targets}[target]
    return _target(loglike,
                   W,
                   features,
                   penalty,
                   **kwargs)



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
    signal = np.sqrt(signal_fac * np.log(p))

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


test_group_lasso()
