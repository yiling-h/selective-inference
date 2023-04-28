from __future__ import division, print_function

import numpy as np
import typing

from scipy.stats import norm as ndist, invgamma
from scipy.linalg import fractional_matrix_power
from scipy.linalg import block_diag

from ..algorithms.barrier_affine import solve_barrier_affine_py
from .selective_MLE_jacobian import mle_inference
from ..base import target_query_Interactspec


class PosteriorAtt(typing.NamedTuple):
    logPosterior: float
    grad_logPosterior: np.ndarray


class posterior(object):
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
        and returns (value of log prior, gradient of log prior)
    dispersion : float, optional
        A dispersion parameter for likelihood.
    solve_args : dict
        Arguments passed to solver of affine barrier problem.
    """

    def __init__(self,
                 query_spec,
                 target_spec,
                 useJacobian,
                 Jacobian_spec,
                 dispersion,
                 prior,
                 solve_args={'tol': 1.e-12}):
        self.query_spec = QS = query_spec
        self.target_spec = TS = target_spec
        self.solve_args = solve_args
        self.useJacobian = useJacobian
        self.Jacobian_spec = Jacobian_spec

        G = mle_inference(query_spec,
                          target_spec,
                          useJacobian,
                          Jacobian_spec,
                          solve_args=solve_args)

        result, self.inverse_info, self.log_ref = G.solve_estimating_eqn()

        self.ntarget = TS.cov_target.shape[0]
        self.nopt = QS.cond_cov.shape[0]

        self.initial_estimate = np.asarray(result['MLE'])
        self.dispersion = dispersion

        ### Note for an informative prior we might want to change this...
        self.prior = prior

        self._get_marginal_parameters()

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

        QS = self.query_spec
        TS = self.target_spec
        if self.useJacobian:
            # C = V.T.dot(QI).dot(L).dot(V)
            # active_dirs
            JS = self.Jacobian_spec

        (prec_marginal,
         linear_coef,
         offset_coef,
         r,
         S,
         prec_target_nosel) = self._get_marginal_parameters()

        sigmasq = sigma ** 2

        target = S.dot(target_parameter) + r

        mean_marginal = linear_coef.dot(target) + offset_coef
        conjugate_marginal = prec_marginal.dot(mean_marginal)

        solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_marginal,
                                 prec_marginal,
                                 QS.observed_soln,
                                 QS.linear_part,
                                 QS.offset,
                                 **self.solve_args)
        if not self.useJacobian:
            val, soln, hess = solver(conjugate_marginal,
                                     prec_marginal,
                                     QS.observed_soln,
                                     QS.linear_part,
                                     QS.offset,
                                     **self.solve_args)
        else:
            val, soln, hess = solve_barrier_affine_jacobian_py(conjugate_marginal,
                                                               prec_marginal,
                                                               QS.observed_soln,
                                                               QS.linear_part,
                                                               QS.offset,
                                                               JS.C,    # for Jacobian
                                                               JS.active_dirs,
                                                               useJacobian=True,
                                                               **self.solve_args)
            log_jacob = jacobian_grad_hess(soln, JS.C, JS.active_dirs)

        assert self.useJacobian == True
        log_normalizer = -val - mean_marginal.T.dot(prec_marginal).dot(mean_marginal) / 2. + log_jacob[0]

        log_lik = -((TS.observed_target - target).T.dot(prec_target_nosel).dot(TS.observed_target - target)) / 2. \
                  - log_normalizer

        grad_lik = S.T.dot(prec_target_nosel.dot(TS.observed_target) - prec_target_nosel.dot(target)
                           - linear_coef.T.dot(prec_marginal.dot(soln) - conjugate_marginal)
                           - linear_coef.T.dot(prec_marginal.dot(hess).dot(log_jacob[1])))

        log_prior, grad_prior = self.prior(target_parameter)

        log_posterior = self.dispersion * (log_lik - self.log_ref) / sigmasq + log_prior
        grad_log_posterior = self.dispersion * grad_lik / sigmasq + grad_prior

        return PosteriorAtt(log_posterior,
                            grad_log_posterior)

    ### Private method

    def _get_marginal_parameters(self):
        """
        This works out the implied covariance
        of optimization varibles as a function
        of randomization as well how to compute
        implied mean as a function of the true parameters.
        """

        QS = self.query_spec
        TS = self.target_spec

        U1, U2, U3, U4, U5 = target_query_Interactspec(QS,
                                                       TS.regress_target_score,
                                                       TS.cov_target)

        prec_target = np.linalg.inv(TS.cov_target)
        cond_precision = np.linalg.inv(QS.cond_cov)

        prec_target_nosel = prec_target + U2 - U3

        _P = -(U1.T.dot(QS.M1.dot(QS.observed_score)) + U2.dot(TS.observed_target))

        bias_target = TS.cov_target.dot(U1.T.dot(-U4.dot(TS.observed_target) +
                                                 QS.M1.dot(QS.opt_linear.dot(QS.cond_mean))) - _P)

        ###set parameters for the marginal distribution of optimization variables

        _Q = np.linalg.inv(prec_target_nosel + U3)
        prec_marginal = cond_precision - U5.T.dot(_Q).dot(U5)
        linear_coef = QS.cond_cov.dot(U5.T)
        offset_coef = QS.cond_mean - linear_coef.dot(TS.observed_target)

        ###set parameters for the marginal distribution of target

        r = np.linalg.inv(prec_target_nosel).dot(prec_target.dot(bias_target))
        S = np.linalg.inv(prec_target_nosel).dot(prec_target)

        return (prec_marginal,
                linear_coef,
                offset_coef,
                r,
                S,
                prec_target_nosel)


### sampling methods

def langevin_sampler(selective_posterior,
                     nsample=2000,
                     nburnin=100,
                     proposal_scale=None,
                     step=1.):
    state = selective_posterior.initial_estimate
    stepsize = 1. / (step * selective_posterior.ntarget)

    if proposal_scale is None:
        proposal_scale = selective_posterior.inverse_info

    sampler = langevin(state,
                       selective_posterior.log_posterior,
                       proposal_scale,
                       stepsize,
                       np.sqrt(selective_posterior.dispersion))

    samples = np.zeros((nsample, selective_posterior.ntarget))

    for i, sample in enumerate(sampler):
        sampler.scaling = np.sqrt(selective_posterior.dispersion)
        samples[i, :] = sample.copy()
        # print("sample ", i, samples[i,:])
        if i == nsample - 1:
            break

    return samples[nburnin:, :]


def gibbs_sampler(selective_posterior,
                  nsample=2000,
                  nburnin=100,
                  proposal_scale=None,
                  step=1.):
    state = selective_posterior.initial_estimate
    stepsize = 1. / (step * selective_posterior.ntarget)

    if proposal_scale is None:
        proposal_scale = selective_posterior.inverse_info

    sampler = langevin(state,
                       selective_posterior.log_posterior,
                       proposal_scale,
                       stepsize,
                       np.sqrt(selective_posterior.dispersion))
    samples = np.zeros((nsample, selective_posterior.ntarget))
    scale_samples = np.zeros(nsample)
    scale_update = np.sqrt(selective_posterior.dispersion)
    for i in range(nsample):
        sample = sampler.__next__()
        samples[i, :] = sample

        import sys
        sys.stderr.write('a: ' + str(0.1 +
                                     selective_posterior.ntarget +
                                     selective_posterior.ntarget / 2) + '\n')
        sys.stderr.write('scale: ' + str(0.1 - ((scale_update ** 2) * sampler.posterior_[0])) + '\n')
        sys.stderr.write('scale_update: ' + str(scale_update) + '\n')
        sys.stderr.write('initpoint: ' + str(sampler.posterior_[0]) + '\n')
        scale_update_sq = invgamma.rvs(a=(0.1 +
                                          selective_posterior.ntarget +
                                          selective_posterior.ntarget / 2),
                                       scale=0.1 - ((scale_update ** 2) * sampler.posterior_.logPosterior),
                                       size=1)
        scale_samples[i] = np.sqrt(scale_update_sq)
        sampler.scaling = np.sqrt(scale_update_sq)

    return samples[nburnin:, :], scale_samples[nburnin:]


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
            self.posterior_ = self.gradient_map(self.state, self.scaling)
            _proposal = self.proposal_sqrt.dot(self._noise.rvs(self._shape))
            candidate = (self.state + self.stepsize * self.proposal_scale.dot(self.posterior_.grad_logPosterior)
                         + np.sqrt(2.) * _proposal * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate, self.scaling)[1])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                break
        return self.state


def target_query_Interactspec(query_spec,
                              regress_target_score,
                              cov_target):
    QS = query_spec
    prec_target = np.linalg.inv(cov_target)

    U1 = regress_target_score.T.dot(prec_target)
    U2 = U1.T.dot(QS.M2.dot(U1))
    U3 = U1.T.dot(QS.M3.dot(U1))
    U4 = QS.M1.dot(QS.opt_linear).dot(QS.cond_cov).dot(QS.opt_linear.T.dot(QS.M1.T.dot(U1)))
    U5 = U1.T.dot(QS.M1.dot(QS.opt_linear))

    return U1, U2, U3, U4, U5

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
        p4 = np.log(1. + 1. / ((con_offset - con_linear.dot(gs)) / scaling)).sum()
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

    hess = np.linalg.inv(precision + barrier_hessian(current))
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