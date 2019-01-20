import numpy as np
import regreg.api as rr
from scipy.stats import norm as ndist

from rpy2.robjects.packages import importr
from rpy2 import robjects

glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()


class query(object):

    def __init__(self, randomization):

        self.randomization = randomization
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self):

        if not self._randomized:
            self.randomized_loss = self.randomization.randomize(self.loss, self.epsilon)
        self._randomized = True

    def randomization_gradient(self, data_state, data_transform, opt_state):
        """
        Randomization derivative at full state.
        """

        # reconstruction of randoimzation omega

        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state) + data_offset
        opt_piece = opt_linear.dot(opt_state) + opt_offset

        # value of the randomization omega

        full_state = (data_piece + opt_piece)

        # gradient of negative log density of randomization at omega

        randomization_derivative = self.randomization.gradient(full_state)

        # chain rule for data, optimization parts

        data_grad = data_linear.T.dot(randomization_derivative)
        opt_grad = opt_linear.T.dot(randomization_derivative)

        return data_grad, opt_grad - self.grad_log_jacobian(opt_state)

    def linear_decomposition(self, target_score_cov, target_cov, observed_target_state):
        """
        Compute out the linear decomposition
        of the score based on the target. This decomposition
        writes the (limiting CLT version) of the data in the score as linear in the
        target and in some independent Gaussian error.
        This second independent piece is conditioned on, resulting
        in a reconstruction of the score as an affine function of the target
        where the offset is the part related to this independent
        Gaussian error.
        """

        target_score_cov = np.atleast_2d(target_score_cov)
        target_cov = np.atleast_2d(target_cov)
        observed_target_state = np.atleast_1d(observed_target_state)

        linear_part = target_score_cov.T.dot(np.linalg.pinv(target_cov))

        offset = self.observed_score_state - linear_part.dot(observed_target_state)

        # now compute the composition of this map with
        # self.score_transform

        score_linear, score_offset = self.score_transform
        composition_linear_part = score_linear.dot(linear_part)

        composition_offset = score_linear.dot(offset) + score_offset

        return (composition_linear_part, composition_offset)

    def reconstruction_map(self, data_state, data_transform, opt_state):

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        # reconstruction of randoimzation omega

        data_state = np.atleast_2d(data_state)
        opt_state = np.atleast_2d(opt_state)

        opt_linear, opt_offset = self.opt_transform
        data_linear, data_offset = data_transform
        data_piece = data_linear.dot(data_state.T) + data_offset[:, None]
        opt_piece = opt_linear.dot(opt_state.T) + opt_offset[:, None]

        # value of the randomization omega

        return (data_piece + opt_piece).T

    def log_density(self, data_state, data_transform, opt_state):

        full_data = self.reconstruction_map(data_state, data_transform, opt_state)
        return self.randomization.log_density(full_data)

    # Abstract methods to be
    # implemented by subclasses

    def grad_log_jacobian(self, opt_state):
        """
        log_jacobian depends only on data through
        Hessian at \bar{\beta}_E which we
        assume is close to Hessian at \bar{\beta}_E^*
        """
        # needs to be implemented for group lasso
        return 0.

    def jacobian(self, opt_state):
        """
        log_jacobian depends only on data through
        Hessian at \bar{\beta}_E which we
        assume is close to Hessian at \bar{\beta}_E^*
        """
        # needs to be implemented for group lasso
        return 1.

    def solve(self):

        raise NotImplementedError('abstract method')

    def setup_sampler(self):
        """
        Setup query to prepare for sampling.
        Should set a few key attributes:
            - observed_score_state
            - num_opt_var
            - observed_opt_state
            - opt_transform
            - score_transform
        """
        raise NotImplementedError('abstract method -- only keyword arguments')

    def projection(self, opt_state):

        raise NotImplementedError('abstract method -- projection of optimization variables')

class M_estimator(query):

    def __init__(self, loss, epsilon, penalty, randomization, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fits the logistic regression to a candidate active set, without penalty.
        Calls the method bootstrap_covariance() to bootstrap the covariance matrix.
        Computes $\bar{\beta}_E$ which is the restricted
        M-estimator (i.e. subject to the constraint $\beta_{-E}=0$).
        Parameters:
        -----------
        active: np.bool
            The active set from fitting the logistic lasso
        solve_args: dict
            Arguments to be passed to regreg solver.
        Returns:
        --------
        None
        Notes:
        ------
        Sets self._beta_unpenalized which will be used in the covariance matrix calculation.
        Also computes Hessian of loss at restricted M-estimator as well as the bootstrap covariance.
        """

        query.__init__(self, randomization)

        (self.loss,
         self.epsilon,
         self.penalty,
         self.randomization,
         self.solve_args) = (loss,
                             epsilon,
                             penalty,
                             randomization,
                             solve_args)

    # Methods needed for subclassing a query

    def solve(self, scaling=1, solve_args={'min_its':20, 'tol':1.e-10}):

        self.randomize()

        (loss,
         randomized_loss,
         epsilon,
         penalty,
         randomization,
         solve_args) = (self.loss,
                        self.randomized_loss,
                        self.epsilon,
                        self.penalty,
                        self.randomization,
                        self.solve_args)

        # initial solution

        problem = rr.simple_problem(randomized_loss, penalty)
        self.initial_soln = problem.solve(**solve_args)

        # find the active groups and their direction vectors
        # as well as unpenalized groups

        groups = np.unique(penalty.groups)
        active_groups = np.zeros(len(groups), np.bool)
        unpenalized_groups = np.zeros(len(groups), np.bool)

        active_directions = []
        active = np.zeros(loss.shape, np.bool)
        unpenalized = np.zeros(loss.shape, np.bool)

        initial_scalings = []

        for i, g in enumerate(groups):
            group = penalty.groups == g
            active_groups[i] = (np.linalg.norm(self.initial_soln[group]) > 1.e-6 * penalty.weights[g]) and (penalty.weights[g] > 0)
            unpenalized_groups[i] = (penalty.weights[g] == 0)
            if active_groups[i]:
                active[group] = True
                z = np.zeros(active.shape, np.float)
                z[group] = self.initial_soln[group] / np.linalg.norm(self.initial_soln[group])
                active_directions.append(z)
                initial_scalings.append(np.linalg.norm(self.initial_soln[group]))
            if unpenalized_groups[i]:
                unpenalized[group] = True

        # solve the restricted problem

        self._overall = active + unpenalized
        self._inactive = ~self._overall
        self._unpenalized = unpenalized
        self._active_directions = np.array(active_directions).T
        self._active_groups = np.array(active_groups, np.bool)
        self._unpenalized_groups = np.array(unpenalized_groups, np.bool)

        self.selection_variable = {'groups':self._active_groups,
                                   'variables':self._overall,
                                   'directions':self._active_directions}

        # initial state for opt variables

        initial_subgrad = -(self.randomized_loss.smooth_objective(self.initial_soln, 'grad') +
                            self.randomized_loss.quadratic.objective(self.initial_soln, 'grad'))
                          # the quadratic of a smooth_atom is not included in computing the smooth_objective

        initial_subgrad = initial_subgrad[self._inactive]
        initial_unpenalized = self.initial_soln[self._unpenalized]
        self.observed_opt_state = np.concatenate([initial_scalings,
                                                  initial_unpenalized,
                                                  initial_subgrad], axis=0)

        # set the _solved bit

        self._solved = True

        # Now setup the pieces for linear decomposition

        (loss,
         epsilon,
         penalty,
         initial_soln,
         overall,
         inactive,
         unpenalized,
         active_groups,
         active_directions) = (self.loss,
                               self.epsilon,
                               self.penalty,
                               self.initial_soln,
                               self._overall,
                               self._inactive,
                               self._unpenalized,
                               self._active_groups,
                               self._active_directions)

        # scaling should be chosen to be Lipschitz constant for gradient of Gaussian part

        # we are implicitly assuming that
        # loss is a pairs model

        _sqrt_scaling = np.sqrt(scaling)

        _beta_unpenalized = restricted_Mest(loss, overall, solve_args=solve_args)

        beta_full = np.zeros(overall.shape)
        beta_full[overall] = _beta_unpenalized
        _hessian = loss.hessian(beta_full)
        self._beta_full = beta_full

        # observed state for score

        self.observed_score_state = np.hstack([_beta_unpenalized * _sqrt_scaling,
                                               -loss.smooth_objective(beta_full, 'grad')[inactive] / _sqrt_scaling])

        # form linear part

        self.num_opt_var = p = loss.shape[0] # shorthand for p

        # (\bar{\beta}_{E \cup U}, N_{-E}, c_E, \beta_U, z_{-E})
        # E for active
        # U for unpenalized
        # -E for inactive

        _opt_linear_term = np.zeros((p, self._active_groups.sum() + unpenalized.sum() + inactive.sum()))
        _score_linear_term = np.zeros((p, p))

        # \bar{\beta}_{E \cup U} piece -- the unpenalized M estimator

        Mest_slice = slice(0, overall.sum())
        _Mest_hessian = _hessian[:,overall]
        _score_linear_term[:,Mest_slice] = -_Mest_hessian / _sqrt_scaling

        # N_{-(E \cup U)} piece -- inactive coordinates of score of M estimator at unpenalized solution

        null_idx = range(overall.sum(), p)
        inactive_idx = np.nonzero(inactive)[0]
        for _i, _n in zip(inactive_idx, null_idx):
            _score_linear_term[_i,_n] = -_sqrt_scaling

        # c_E piece

        scaling_slice = slice(0, active_groups.sum())
        if len(active_directions)==0:
            _opt_hessian=0
        else:
            _opt_hessian = (_hessian + epsilon * np.identity(p)).dot(active_directions)
        _opt_linear_term[:,scaling_slice] = _opt_hessian / _sqrt_scaling

        self.observed_opt_state[scaling_slice] *= _sqrt_scaling

        # beta_U piece

        unpenalized_slice = slice(active_groups.sum(), active_groups.sum() + unpenalized.sum())
        unpenalized_directions = np.identity(p)[:,unpenalized]
        if unpenalized.sum():
            _opt_linear_term[:,unpenalized_slice] = (_hessian + epsilon * np.identity(p)).dot(unpenalized_directions) / _sqrt_scaling

        self.observed_opt_state[unpenalized_slice] *= _sqrt_scaling

        # subgrad piece

        subgrad_idx = range(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        subgrad_slice = slice(active_groups.sum() + unpenalized.sum(), active_groups.sum() + inactive.sum() + unpenalized.sum())
        for _i, _s in zip(inactive_idx, subgrad_idx):
            _opt_linear_term[_i,_s] = _sqrt_scaling

        self.observed_opt_state[subgrad_slice] /= _sqrt_scaling

        # form affine part

        _opt_affine_term = np.zeros(p)
        idx = 0
        groups = np.unique(penalty.groups)
        for i, g in enumerate(groups):
            if active_groups[i]:
                group = penalty.groups == g
                _opt_affine_term[group] = active_directions[:,idx][group] * penalty.weights[g]
                idx += 1

        # two transforms that encode score and optimization
        # variable roles

        self.opt_transform = (_opt_linear_term, _opt_affine_term)
        self.score_transform = (_score_linear_term, np.zeros(_score_linear_term.shape[0]))

        # later, we will modify `score_transform`
        # in `linear_decomposition`

        # now store everything needed for the projections
        # the projection acts only on the optimization
        # variables

        self.scaling_slice = scaling_slice

        # weights are scaled here because the linear terms scales them by scaling

        new_groups = penalty.groups[inactive]
        new_weights = dict([(g, penalty.weights[g] / _sqrt_scaling) for g in penalty.weights.keys() if g in np.unique(new_groups)])

        # we form a dual group lasso object
        # to do the projection

        self.group_lasso_dual = rr.group_lasso_dual(new_groups, weights=new_weights, bound=1.)
        self.subgrad_slice = subgrad_slice

        self._setup = True

    def setup_sampler(self, scaling=1, solve_args={'min_its':20, 'tol':1.e-10}):
        pass

    def projection(self, opt_state):
        """
        Full projection for Langevin.
        The state here will be only the state of the optimization variables.
        """

        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')


        if ('subgradient' not in self.selection_variable and
            'scaling' not in self.selection_variable): # have not conditioned on any thing else
            new_state = opt_state.copy() # not really necessary to copy
            new_state[self.scaling_slice] = np.maximum(opt_state[self.scaling_slice], 0)
            new_state[self.subgrad_slice] = self.group_lasso_dual.bound_prox(opt_state[self.subgrad_slice])
        elif ('subgradient' not in self.selection_variable and
              'scaling' in self.selection_variable): # conditioned on the initial scalings
                                                     # only the subgradient in opt_state
            new_state = self.group_lasso_dual.bound_prox(opt_state)
        elif ('subgradient' in self.selection_variable and
              'scaling' not in self.selection_variable): # conditioned on the subgradient
                                                         # only the scaling in opt_state
            new_state = np.maximum(opt_state, 0)
        else:
            new_state = opt_state
        return new_state

    # optional things to condition on

    def condition_on_subgradient(self):
        """
        Maybe we should allow subgradients of only some variables...
        """
        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        opt_linear, opt_offset = self.opt_transform

        new_offset = opt_linear[:,self.subgrad_slice].dot(self.observed_opt_state[self.subgrad_slice]) + opt_offset
        new_linear = opt_linear[:,self.scaling_slice]

        self.opt_transform = (new_linear, new_offset)

        # for group LASSO this should not induce a bigger jacobian as
        # the subgradients are in the interior of a ball
        self.selection_variable['subgradient'] = self.observed_opt_state[self.subgrad_slice]

        # reset variables

        self.observed_opt_state = self.observed_opt_state[self.scaling_slice]
        self.scaling_slice = slice(None, None, None)
        self.subgrad_slice = np.zeros(new_linear.shape[1], np.bool)
        self.num_opt_var = new_linear.shape[1]

    def condition_on_scalings(self):
        """
        Maybe we should allow subgradients of only some variables...
        """
        if not self._setup:
            raise ValueError('setup_sampler should be called before using this function')

        opt_linear, opt_offset = self.opt_transform

        new_offset = opt_linear[:,self.scaling_slice].dot(self.observed_opt_state[self.scaling_slice]) + opt_offset
        new_linear = opt_linear[:,self.subgrad_slice]

        self.opt_transform = (new_linear, new_offset)

        # for group LASSO this will induce a bigger jacobian
        self.selection_variable['scalings'] = self.observed_opt_state[self.scaling_slice]

        # reset slices

        self.observed_opt_state = self.observed_opt_state[self.subgrad_slice]
        self.subgrad_slice = slice(None, None, None)
        self.scaling_slice = np.zeros(new_linear.shape[1], np.bool)
        self.num_opt_var = new_linear.shape[1]

def restricted_Mest(Mest_loss, active, solve_args={'min_its':50, 'tol':1.e-10}):

    X, Y = Mest_loss.data

    if Mest_loss._is_transform:
        raise NotImplementedError('to fit restricted model, X must be an ndarray or scipy.sparse; general transforms not implemented')
    X_restricted = X[:,active]
    loss_restricted = rr.affine_smooth(Mest_loss.saturated_loss, X_restricted)
    beta_E = loss_restricted.solve(**solve_args)

    return beta_E

def naive_confidence_intervals(target, observed, alpha=0.1):
    """
    Compute naive Gaussian based confidence
    intervals for target.
    Parameters
    ----------
    target : `targeted_sampler`
    observed : np.float
        A vector of observed data of shape `target.shape`
    alpha : float (optional)
        1 - confidence level.
    Returns
    -------
    intervals : np.float
        Gaussian based confidence intervals.
    """
    quantile = - ndist.ppf(alpha/float(2))
    LU = np.zeros((2, target.shape[0]))
    for j in range(target.shape[0]):
        sigma = np.sqrt(target.target_cov[j, j])
        LU[0,j] = observed[j] - sigma * quantile
        LU[1,j] = observed[j] + sigma * quantile
    return LU.T

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)

                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_minCV = out$lambda.min

                coef = coef(out, s = "lambda.min")
                linear.fit = lm(y~ X[, which(coef>0.001)-1])
                sigma_est = summary(linear.fit)$sigma
                return(sigma_est)
                }''')

    sigma_cv_R = robjects.globalenv['glmnet_cv']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    sigma_est = sigma_cv_R(r_X, r_y)
    return sigma_est


def BH_q(p_value, level):
    m = p_value.shape[0]
    p_sorted = np.sort(p_value)
    indices = np.arange(m)
    indices_order = np.argsort(p_value)

    if np.any(p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= np.zeros(m)):
        order_sig = np.max(indices[p_sorted - np.true_divide(level * (np.arange(m) + 1.), m) <= 0])
        sig_pvalues = indices_order[:(order_sig + 1)]
        return p_sorted[:(order_sig + 1)], sig_pvalues

    else:
        return None