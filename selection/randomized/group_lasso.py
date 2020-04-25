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

def test_group_lasso(n=200,
                     p=50,
                     signal_fac=3,
                     s=5,
                     sigma=3,
                     target='selected',
                     rho=0.4,
                     randomizer_scale= 1.):


    inst = gaussian_instance
    signal = np.sqrt(signal_fac * np.log(p))

    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      s=s,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]


    n, p = X.shape

    sigma_ = np.std(Y)

    groups = np.floor(np.arange(p)/3).astype(np.int)
    weights = dict([(i, sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    conv = group_lasso.gaussian(X,
                                Y,
                                groups,
                                weights,
                                randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()
    nonzero = conv.selection_variable['directions'].keys()
    print("check ", nonzero)

test_group_lasso(n=200,
                 p=50,
                 signal_fac=3,
                 s=3,
                 sigma=1.,
                 target='selected',
                 rho=0.4,
                 randomizer_scale= 1.)
