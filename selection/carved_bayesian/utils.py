import numpy as np
from scipy.stats import norm as ndist
import regreg.api as rr


class projected_langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 projection_map,
                 stepsize):

        (self.state,
         self.gradient_map,
         self.projection_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           projection_map,
                           stepsize)
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)

    def __iter__(self):
        return self

    def next(self):
        while True:
            proj_arg = (self.state + 0.5 * self.stepsize * self.gradient_map(self.state)
                        + self._noise.rvs(self._shape) * self._sqrt_step)
            candidate = self.projection_map(proj_arg)
            if not np.all(np.isfinite(self.gradient_map(candidate))):
                print(candidate, self._sqrt_step)
                self._sqrt_step *= 0.8
            else:
                self.state[:] = candidate
                break

class nonnegative_softmax_scaled(rr.smooth_atom):
    """
    The nonnegative softmax objective
    .. math::
         \mu \mapsto
         \sum_{i=1}^{m} \log \left(1 +
         \frac{1}{\mu_i} \right)
    """

    objective_template = r"""\text{nonneg_softmax}\left(%(var)s\right)"""

    def __init__(self,
                 shape,
                 barrier_scale=1.,
                 coef=1.,
                 offset=None,
                 quadratic=None,
                 initial=None):

        rr.smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        # a feasible point
        self.coefs[:] = np.ones(shape)
        self.barrier_scale = barrier_scale

    def smooth_objective(self, mean_param, mode='both', check_feasibility=False):
        """
        Evaluate the smooth objective, computing its value, gradient or both.
        Parameters
        ----------
        mean_param : ndarray
            The current parameter values.
        mode : str
            One of ['func', 'grad', 'both'].
        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `mean_param` is not
            in the domain.
        Returns
        -------
        If `mode` is 'func' returns just the objective value
        at `mean_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """

        slack = self.apply_offset(mean_param)

        if mode in ['both', 'func']:
            if np.all(slack > 0):
                f = self.scale(np.log((slack + self.barrier_scale) / slack).sum())
            else:
                f = np.inf
        if mode in ['both', 'grad']:
            g = self.scale(1. / (slack + self.barrier_scale) - 1. / slack)

        if mode == 'both':
            return f, g
        elif mode == 'grad':
            return g
        elif mode == 'func':
            return f
        else:
            raise ValueError("mode incorrectly specified")


class neg_log_cube_probability(rr.smooth_atom):
    def __init__(self,
                 q, #equals p - E in our case
                 lagrange,
                 randomization_scale = 1., #equals the randomization variance in our case
                 coef=1.,
                 offset=None,
                 quadratic=None):

        self.randomization_scale = randomization_scale
        self.lagrange = lagrange
        self.q = q

        rr.smooth_atom.__init__(self,
                                (self.q,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=None,
                                coef=coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        arg_u = (arg + self.lagrange) / self.randomization_scale
        arg_l = (arg - self.lagrange) / self.randomization_scale
        prod_arg = np.exp(-(2. * self.lagrange * arg) / (self.randomization_scale ** 2))
        neg_prod_arg = np.exp((2. * self.lagrange * arg) / (self.randomization_scale ** 2))
        cube_prob = ndist.cdf(arg_u) - ndist.cdf(arg_l)
        log_cube_prob = -np.log(cube_prob).sum()
        threshold = 10 ** -10
        indicator = np.zeros(self.q, bool)
        indicator[(cube_prob > threshold)] = 1
        positive_arg = np.zeros(self.q, bool)
        positive_arg[(arg > 0)] = 1
        pos_index = np.logical_and(positive_arg, ~indicator)
        neg_index = np.logical_and(~positive_arg, ~indicator)
        log_cube_grad = np.zeros(self.q)
        log_cube_grad[indicator] = (np.true_divide(-ndist.pdf(arg_u[indicator]) + ndist.pdf(arg_l[indicator]),
                                                   cube_prob[indicator])) / self.randomization_scale

        log_cube_grad[pos_index] = ((-1. + prod_arg[pos_index]) /
                                    ((prod_arg[pos_index] / arg_u[pos_index]) -
                                     (1. / arg_l[pos_index]))) / self.randomization_scale

        log_cube_grad[neg_index] = ((arg_u[neg_index] - (arg_l[neg_index] * neg_prod_arg[neg_index]))
                                    / self.randomization_scale) / (1. - neg_prod_arg[neg_index])

        if mode == 'func':
            return self.scale(log_cube_prob)
        elif mode == 'grad':
            return self.scale(log_cube_grad)
        elif mode == 'both':
            return self.scale(log_cube_prob), self.scale(log_cube_grad)
        else:
            raise ValueError("mode incorrectly specified")

class log_likelihood(rr.smooth_atom):

    def __init__(self,
                 mean,
                 Sigma,
                 m,
                 coef=1.,
                 offset=None,
                 quadratic=None):

        initial = np.zeros(m)

        self.mean = mean

        self.Sigma = Sigma

        rr.smooth_atom.__init__(self,
                                (m,),
                                offset=offset,
                                quadratic=quadratic,
                                initial=initial,
                                coef=coef)

        self.coefs[:] = initial

    def smooth_objective(self, arg, mode='both', check_feasibility=False, tol=1.e-6):

        arg = self.apply_offset(arg)

        f = ((arg-self.mean).T.dot(np.linalg.inv(self.Sigma)).dot(arg-self.mean))/2.

        g = (np.linalg.inv(self.Sigma)).dot(arg-self.mean)

        if mode == 'func':
            return f

        elif mode == 'grad':
            return g

        elif mode == 'both':
            return f, g

        else:
            raise ValueError('mode incorrectly specified')