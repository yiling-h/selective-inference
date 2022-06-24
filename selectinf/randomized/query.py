import functools
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm as ndist
from scipy.optimize import bisect

from regreg.affine import power_L
import regreg.api as rr

from ..distributions.api import discrete_family
from ..constraints.affine import (sample_from_constraints,
                                  constraints)
from ..algorithms.barrier_affine import solve_barrier_affine_py
from ..base import (selected_targets,
                    full_targets,
                    debiased_targets)

from .posterior_inference import posterior
from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from .approx_reference import approximate_grid_inference

class query(object):
    r"""
    This class is the base of randomized selective inference
    based on convex programs.
    The main mechanism is to take an initial penalized program
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B)
    and add a randomization and small ridge term yielding
    .. math::
        \text{minimize}_B \ell(B) + {\cal P}(B) -
        \langle \omega, B \rangle + \frac{\epsilon}{2} \|B\|^2_2
    """

    def __init__(self, randomization, perturb=None):

        """
        Parameters
        ----------
        randomization : `selection.randomized.randomization.randomization`
            Instance of a randomization scheme.
            Describes the law of $\omega$.
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """
        self.randomization = randomization
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    # Methods reused by subclasses

    def randomize(self, perturb=None):

        """
        The actual randomization step.
        Parameters
        ----------
        perturb : ndarray, optional
            Value of randomization vector, an instance of $\omega$.
        """

        if not self._randomized:
            (self.randomized_loss,
             self._initial_omega) = self.randomization.randomize(self.loss,
                                                                 self.epsilon,
                                                                 perturb=perturb)
        self._randomized = True

    def get_sampler(self):
        if hasattr(self, "_sampler"):
            return self._sampler

    def set_sampler(self, sampler):
        self._sampler = sampler

    sampler = property(get_sampler, set_sampler, doc='Sampler of optimization (augmented) variables.')

    # implemented by subclasses

    def solve(self):

        raise NotImplementedError('abstract method')


class gaussian_query(query):
    useC = True

    """
    A class with Gaussian perturbation to the objective -- 
    easy to apply CLT to such things
    """

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

    # Private methods

    def _setup_sampler(self,
                       linear_part,
                       offset,
                       opt_linear,
                       observed_subgrad,
                       dispersion=1):

        A, b = linear_part, offset
        if not np.all(A.dot(self.observed_opt_state) - b <= 0):
            raise ValueError('constraints not satisfied')

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

        # cond_mean = A = -(Sigma_bar Q_s' Sigma_w^{-1}) (P_s + r_s),
        # where P_s = -X'X_E = 'observed_score_state' - beta_s^perp,
        # and   r_s = 'observed_subgrad' + beta_s^perp
        # cond_cov = Sigma_bar^{-1}
        # cond_precision = Sigma_bar
        # regress_opt = -(Sigma_bar Q_s' Sigma_w^{-1})
        # M1 = X'X Sigma_w^{-1} * dispersion
        # M2 = dispersion * X'X Sigma_w^{-1} X'X
        # M3 = dispersion^2 * X'X Sigma_w^{-1} (Q_s Sigma_bar Q_s') Sigma_w^{-1} X'X
        (cond_mean,
         cond_cov,
         cond_precision,
         regress_opt,
         M1,
         M2,
         M3) = self._setup_implied_gaussian(opt_linear,
                                            observed_subgrad,
                                            dispersion=dispersion)

        def log_density(regress_opt, u, cond_prec, opt, score):  # u == subgrad
            if score.ndim == 1:
                mean_term = regress_opt.dot(score.T + u).T
            else:
                mean_term = regress_opt.dot(score.T + u[:, None]).T
            arg = opt - mean_term
            return - 0.5 * np.sum(arg * cond_prec.dot(arg.T).T, 1)

        log_density = functools.partial(log_density,
                                        regress_opt,
                                        observed_subgrad,
                                        cond_precision)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov

        # In LASSO, A = U = - diag(sign(o1)), b = v = 0_E
        affine_con = constraints(A,
                                 b,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        self.sampler = affine_gaussian_sampler(affine_con,
                                               self.observed_opt_state,
                                               self.observed_score_state,
                                               log_density,
                                               regress_opt,  # not needed?
                                               observed_subgrad,
                                               opt_linear,  # L
                                               M1,
                                               M2,
                                               M3,
                                               selection_info=self.selection_variable,
                                               useC=self.useC)

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):
        # For LASSO, opt_linear = Q_s
        # cov_rand = Sigma_w, prec = Sigma_w^{-1}
        cov_rand, prec = self.randomizer.cov_prec

        # For LASSO, '_unscaled_cov_score' = '_hessian' = X'X
        if np.asarray(prec).shape in [(), (0,)]:
            prod_score_prec_unnorm = self._unscaled_cov_score * prec
        else:
            prod_score_prec_unnorm = self._unscaled_cov_score.dot(prec)

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T) * prec
        else:
            # cond_precision = Sigma_bar^{-1}
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            # cond_cov = Sigma_bar
            cond_cov = np.linalg.inv(cond_precision)
            # regress_opt = -(Sigma_bar Q_s' Sigma_w^{-1})
            regress_opt = -cond_cov.dot(opt_linear.T).dot(prec)

        # regress_opt is regression coefficient of opt onto score + u...
        # cond_mean = A beta_hat_s + b = -(Sigma_bar Q_s' Sigma_w^{-1}) (P_s beta_hat_s + r_s),
        # where P_s = -X'X_E, P_s beta_hat_s = 'observed_score_state' - beta_s^perp,
        # and   r_s = 'observed_subgrad' + beta_s^perp
        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        # M1 = X'X Sigma_w^{-1} * dispersion
        M1 = prod_score_prec_unnorm * dispersion
        # M2 = dispersion^2 * X'X Sigma_w^{-1} X'X
        M2 = M1.dot(cov_rand).dot(M1.T)
        # M3 = dispersion^2 * X'X Sigma_w^{-1} (Q_s Sigma_bar Q_s') Sigma_w^{-1} X'X
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                regress_opt,
                M1,
                M2,
                M3)

    def summary(self,
                target_spec,
                opt_sample=None,
                target_sample=None,
                parameter=None,
                level=0.9,
                ndraw=10000,
                burnin=2000,
                compute_intervals=False):
        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated regression coefficient of target on score.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        compute_intervals : bool
            Compute confidence intervals?
        """

        if parameter is None:
            parameter = np.zeros_like(target_spec.observed_target)

        if opt_sample is None:
            opt_sample, logW = self.sampler.sample(ndraw, burnin)
        else:
            if len(opt_sample) == 1:  # only a sample, so weights are 1s
                opt_sample = opt_sample[0]
                logW = np.zeros(ndraw)
            else:
                opt_sample, logW = opt_sample
            ndraw = opt_sample.shape[0]

        pivots = self.sampler.coefficient_pvalues(target_spec.observed_target,
                                                  target_spec.cov_target,
                                                  target_spec.regress_target_score,
                                                  parameter=parameter,
                                                  sample=(opt_sample, logW),
                                                  normal_sample=target_sample,
                                                  alternatives=target_spec.alternatives)

        if not np.all(parameter == 0):
            pvalues = self.sampler.coefficient_pvalues(target_spec.observed_target,
                                                       target_spec.cov_target,
                                                       target_spec.regress_target_score,
                                                       parameter=np.zeros_like(parameter),
                                                       sample=(opt_sample, logW),
                                                       normal_sample=target_sample,
                                                       alternatives=target_spec.alternatives)
        else:
            pvalues = pivots

        result = pd.DataFrame({'target': target_spec.observed_target,
                               'pvalue': pvalues})

        if compute_intervals:
            MLE = self.selective_MLE(target_spec)[0]
            MLE_intervals = np.asarray(MLE[['lower_confidence', 'upper_confidence']])

            intervals = self.sampler.confidence_intervals(
                target_spec.observed_target,
                target_spec.cov_target,
                target_spec.regress_target_score,
                sample=(opt_sample, logW),
                normal_sample=target_sample,
                initial_guess=MLE_intervals,
                level=level)

            result.insert(2, 'lower_confidence', intervals[:, 0])
            result.insert(3, 'upper_confidence', intervals[:, 1])

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

    def selective_MLE(self,
                      target_spec,
                      level=0.9,
                      solve_args={'tol': 1.e-12}):
        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        level : float, optional
            Confidence level.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        return self.sampler.selective_MLE(target_spec,
                                          self.observed_opt_state,
                                          solve_args=solve_args,
                                          level=level)

    def posterior(self,
                  target_spec,
                  dispersion=1,
                  prior=None,
                  solve_args={'tol': 1.e-12}):
        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        prior : callable
            A callable object that takes a single argument
            `parameter` of the same shape as `observed_target`
            and returns (value of log prior, gradient of log prior)
        dispersion : float, optional
            Dispersion parameter for log-likelihood.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        if prior is None:
            Di = 1. / (200 * np.diag(target_spec.cov_target))

            def prior(target_parameter):
                grad_prior = -target_parameter * Di
                log_prior = -0.5 * np.sum(target_parameter ** 2 * Di)
                return log_prior, grad_prior

        return posterior(self,
                         target_spec,
                         dispersion,
                         prior,
                         solve_args=solve_args)

    def approximate_grid_inference(self,
                                   target_spec,
                                   solve_args={'tol': 1.e-12},
                                   useIP=True):

        """
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        solve_args : dict, optional
            Arguments passed to solver.
        """

        G = approximate_grid_inference(self,
                                       target_spec,
                                       solve_args=solve_args,
                                       useIP=useIP)
        return G.summary(alternatives=target_spec.alternatives)


class multiple_queries(object):
    '''
    Combine several queries of a given data
    through randomized algorithms.
    '''

    def __init__(self, objectives):
        '''
        Parameters
        ----------
        objectives : sequence
           A sequences of randomized objective functions.
        Notes
        -----
        Each element of `objectives` must
        have a `setup_sampler` method that returns
        a description of the distribution of the
        data implicated in the objective function,
        typically through the score or gradient
        of the objective function.
        These descriptions are passed to a function
        `form_covariances` to linearly decompose
        each score in terms of a target
        and an asymptotically independent piece.
        Returns
        -------
        None
        '''

        self.objectives = objectives

    def fit(self):
        for objective in self.objectives:
            if not objective._setup:
                objective.fit()

    def summary(self,
                target_specs,
                # a sequence of target_specs
                # objects in theory all cov_target
                # should be about the same. as should the observed_target
                alternatives=None,
                parameter=None,
                level=0.9,
                ndraw=5000,
                burnin=2000,
                compute_intervals=False):

        """
        Produce p-values and confidence intervals for targets
        of model including selected features
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        parameter : np.array
            Hypothesized value for parameter -- defaults to 0.
        level : float
            Confidence level.
        ndraw : int (optional)
            Defaults to 1000.
        burnin : int (optional)
            Defaults to 1000.
        compute_intervals : bool
            Compute confidence intervals?
        """

        observed_target = target_specs[0].observed_target
        alternatives = target_specs[0].alternatives
        
        if parameter is None:
            parameter = np.zeros_like(observed_target)

        if alternatives is None:
            alternatives = ['twosided'] * observed_target.shape[0]

        if len(self.objectives) != len(target_specs):
            raise ValueError("number of objectives and sampling cov infos do not match")

        self.opt_sampling_info = []
        for i in range(len(self.objectives)):
            if target_specs[i].cov_target is None or target_specs[i].regress_target_score is None:
                raise ValueError("did not input target and score covariance info")
            opt_sample, opt_logW = self.objectives[i].sampler.sample(ndraw, burnin)
            self.opt_sampling_info.append((self.objectives[i].sampler,
                                           opt_sample,
                                           opt_logW,
                                           target_specs[i].cov_target,
                                           target_specs[i].regress_target_score))

        pivots = self.coefficient_pvalues(observed_target,
                                          parameter=parameter,
                                          alternatives=alternatives)

        if not np.all(parameter == 0):
            pvalues = self.coefficient_pvalues(observed_target,
                                               parameter=np.zeros_like(observed_target),
                                               alternatives=alternatives)
        else:
            pvalues = pivots

        intervals = None
        if compute_intervals:
            intervals = self.confidence_intervals(observed_target,
                                                  level)

        result = pd.DataFrame({'target': observed_target,
                               'pvalue': pvalues,
                               'lower_confidence': intervals[:, 0],
                               'upper_confidence': intervals[:, 1]})

        if not np.all(parameter == 0):
            result.insert(4, 'pivot', pivots)
            result.insert(5, 'parameter', parameter)

        return result

    def coefficient_pvalues(self,
                            observed_target,
                            parameter=None,
                            sample_args=(),
                            alternatives=None):

        '''
        Construct selective p-values
        for each parameter of the target.
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        parameter : ndarray (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.
        sample_args : sequence
           Arguments to `self.sample` if sample is not found
           for a given objective.
        alternatives : [str], optional
            Sequence of strings describing the alternatives,
            should be values of ['twosided', 'less', 'greater']
        Returns
        -------
        pvalues : ndarray
        '''

        for i in range(len(self.objectives)):
            if self.opt_sampling_info[i][1] is None:
                _sample, _logW = self.objectives[i].sampler.sample(*sample_args)
                self.opt_sampling_info[i][1] = _sample
                self.opt_sampling_info[i][2] = _logW

        ndraw = self.opt_sampling_info[0][1].shape[0]  # nsample for normal samples taken from the 1st objective

        _intervals = optimization_intervals(self.opt_sampling_info,
                                            observed_target,
                                            ndraw)

        pvals = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            pvals.append(_intervals.pivot(keep, candidate=parameter[i], alternative=alternatives[i]))

        return np.array(pvals)

    def confidence_intervals(self,
                             target_specs,
                             sample_args=(),
                             level=0.9):

        '''
        Construct selective confidence intervals
        for each parameter of the target.
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        sample_args : sequence
           Arguments to `self.sample` if sample is not found
           for a given objective.
        level : float
            Confidence level.
        Returns
        -------
        limits : ndarray
            Confidence intervals for each target.
        '''

        for i in range(len(self.objectives)):
            if self.opt_sampling_info[i][1] is None:
                _sample, _logW = self.objectives[i].sampler.sample(*sample_args)
                self.opt_sampling_info[i][1] = _sample
                self.opt_sampling_info[i][2] = _logW

        ndraw = self.opt_sampling_info[0][1].shape[0]  # nsample for normal samples taken from the 1st objective

        _intervals = optimization_intervals(self.opt_sampling_info,
                                            observed_target,
                                            ndraw)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            limits.append(_intervals.confidence_interval(keep, level=level))

        return np.array(limits)


class optimization_sampler(object):

    def __init__(self):
        raise NotImplementedError("abstract method")

    def sample(self):
        raise NotImplementedError("abstract method")

    def log_cond_density(self,
                         opt_sample,
                         target_sample,
                         transform=None):
        """
        Density of opt_sample | target_sample
        """
        raise NotImplementedError("abstract method")

    def hypothesis_test(self,
                        test_stat,
                        observed_value,
                        cov_target,
                        score_cov,
                        sample_args=(),
                        sample=None,
                        parameter=0,
                        alternative='twosided'):

        '''
        Sample `target` from selective density
        using sampler with
        gradient map `self.gradient` and
        projection map `self.projection`.
        Parameters
        ----------
        test_stat : callable
           Test statistic to evaluate on sample from
           selective distribution.
        observed_value : float
           Observed value of test statistic.
           Used in p-value calculation.
        sample_args : sequence
           Arguments to `self.sample` if sample is None.
        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc. If not None,
           `ndraw, burnin, stepsize` are ignored.
        parameter : np.float (optional)
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalue : float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        if sample is None:
            sample, logW = self.sample(*sample_args)
            sample = np.atleast_2d(sample)

        if parameter is None:
            parameter = self.reference

        sample_test_stat = np.squeeze(np.array([test_stat(x) for x in sample]))

        target_inv_cov = np.linalg.inv(cov_target)
        delta = target_inv_cov.dot(parameter - self.reference)
        W = np.exp(sample.dot(delta) + logW)

        family = discrete_family(sample_test_stat, W)
        pval = family.cdf(0, observed_value)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        else:
            return 2 * min(pval, 1 - pval)

    def confidence_intervals(self,
                             observed_target,
                             cov_target,
                             score_cov,
                             sample_args=(),
                             sample=None,
                             normal_sample=None,
                             level=0.9,
                             initial_guess=None):
        '''
        Parameters
        ----------

        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.
        sample_args : sequence
           Arguments to `self.sample` if sample is None.
        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.
        level : float (optional)
            Specify the
            confidence level.
        initial_guess : np.float
            Initial guesses at upper and lower limits, optional.
        Notes
        -----
        Construct selective confidence intervals
        for each parameter of the target.
        Returns
        -------
        intervals : [(float, float)]
            List of confidence intervals.
        '''

        if sample is None:
            sample, logW = self.sample(*sample_args)
            sample = np.vstack([sample] * 5)  # why times 5?
            logW = np.hstack([logW] * 5)
        else:
            sample, logW = sample

        ndraw = sample.shape[0]

        _intervals = optimization_intervals([(self,
                                              sample,
                                              logW,
                                              cov_target,
                                              score_cov)],
                                            observed_target,
                                            ndraw,
                                            normal_sample=normal_sample)

        limits = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            if initial_guess is None:
                l, u = _intervals.confidence_interval(keep, level=level)
            else:
                l, u = _intervals.confidence_interval(keep, level=level,
                                                      guess=initial_guess[i])
            limits.append((l, u))

        return np.array(limits)

    def coefficient_pvalues(self,
                            observed_target,
                            cov_target,
                            score_cov,
                            parameter=None,
                            sample_args=(),
                            sample=None,
                            normal_sample=None,
                            alternatives=None):
        '''
        Construct selective p-values
        for each parameter of the target.
        Parameters
        ----------
        observed : np.float
            A vector of parameters with shape `self.shape`,
            representing coordinates of the target.
        parameter : np.float (optional)
            A vector of parameters with shape `self.shape`
            at which to evaluate p-values. Defaults
            to `np.zeros(self.shape)`.
        sample_args : sequence
           Arguments to `self.sample` if sample is None.
        sample : np.array (optional)
           If not None, assumed to be a sample of shape (-1,) + `self.shape`
           representing a sample of the target from parameters `self.reference`.
           Allows reuse of the same sample for construction of confidence
           intervals, hypothesis tests, etc.
        alternatives : list of ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalues : np.float
        '''

        if alternatives is None:
            alternatives = ['twosided'] * observed_target.shape[0]

        if sample is None:
            sample, logW = self.sample(*sample_args)
        else:
            sample, logW = sample
            ndraw = sample.shape[0]

        if parameter is None:
            parameter = np.zeros(observed_target.shape[0])

        _intervals = optimization_intervals([(self,
                                              sample,
                                              logW,
                                              cov_target,
                                              score_cov)],
                                            observed_target,
                                            ndraw,
                                            normal_sample=normal_sample)
        pvals = []

        for i in range(observed_target.shape[0]):
            keep = np.zeros_like(observed_target)
            keep[i] = 1.
            pvals.append(_intervals.pivot(keep,
                                          candidate=parameter[i],
                                          alternative=alternatives[i]))

        return np.array(pvals)

    def _reconstruct_score_from_target(self,
                                       target_sample,
                                       transform=None):
        if transform is not None:
            direction, nuisance = transform
            score_sample = (np.multiply.outer(target_sample,
                                              direction) +
                            nuisance[None, :])
        else:
            score_sample = target_sample
        return score_sample


class affine_gaussian_sampler(optimization_sampler):
    '''
    Sample from an affine truncated Gaussian
    '''

    def __init__(self,
                 affine_con,
                 initial_point,
                 observed_score_state,
                 log_cond_density,
                 regress_opt,
                 observed_subgrad,
                 opt_linear,
                 M1,
                 M2,
                 M3,
                 selection_info=None,
                 useC=False):

        '''
        Parameters
        ----------
        affine_con : `selection.constraints.affine.constraints`
             Affine constraints
        initial_point : ndarray
             Feasible point for affine constraints.
        observed_score_state : ndarray
             Observed score of convex loss (slightly modified).
             Essentially (asymptotically) equivalent
             to $\nabla \ell(\beta^*) +
             Q(\beta^*)\beta^*$ where $\beta^*$ is population
             minimizer. For linear regression, it is always
             $-X^Ty$.
        log_cond_density : callable
             Density of optimization variables given score
        regress_opt: ndarray
             Regression coefficient of opt on to score
        observed_subgrad : ndarray
        selection_info : optional
             Function of optimization variables that
             will be conditioned on.
        useC : bool, optional
            Use python or C solver.

        '''

        self.affine_con = affine_con

        self.covariance = self.affine_con.covariance
        self.mean = self.affine_con.mean

        self.initial_point = initial_point
        self.observed_score_state = observed_score_state
        self.selection_info = selection_info
        self._log_cond_density = log_cond_density
        self.regress_opt = regress_opt
        self.observed_subgrad = observed_subgrad
        self.useC = useC
        self.opt_linear = opt_linear
        self.M1, self.M2, self.M3 = M1, M2, M3

    def log_cond_density(self,
                         opt_sample,
                         target_sample,
                         transform=None):

        if transform is not None:
            direction, nuisance = transform
            return self._log_density_ray(0,  # candidate
                                         # has been added to
                                         # target
                                         direction,
                                         nuisance,
                                         target_sample,
                                         opt_sample)
        else:
            # target must be in score coordinates
            score_sample = target_sample

            # probably should switch
            # order of signature
            return self._log_cond_density(opt_sample,
                                          score_sample)

    def sample(self, ndraw, burnin):
        '''
        Sample `target` from selective density
        using projected Langevin sampler with
        gradient map `self.gradient` and
        projection map `self.projection`.
        Parameters
        ----------
        ndraw : int
           How long a chain to return?
        burnin : int
           How many samples to discard?
        '''

        _sample = sample_from_constraints(self.affine_con,
                                          self.initial_point,
                                          ndraw=ndraw,
                                          burnin=burnin)
        return _sample, np.zeros(_sample.shape[0])

    def selective_MLE(self,
                      target_spec,
                      # initial (observed) value of optimization variables --
                      # used as a feasible point.
                      # precise value used only for independent estimator
                      observed_soln,
                      solve_args={'tol': 1.e-12},
                      level=0.9):
        """
        Selective MLE based on approximation of
        CGF.
        Parameters
        ----------
        observed_target : ndarray
            Observed estimate of target.
        cov_target : ndarray
            Estimated covaraince of target.
        regress_target_score : ndarray
            Estimated covariance of target and score of randomized query.
        observed_soln : ndarray
            Feasible point for optimization problem.
        level : float, optional
            Confidence level.
        solve_args : dict, optional
            Arguments passed to solver.
        """

        return selective_MLE(target_spec,
                             observed_soln,
                             self.mean,
                             self.covariance,
                             self.affine_con.linear_part,
                             self.affine_con.offset,
                             self.opt_linear,
                             self.M1,
                             self.M2,
                             self.M3,
                             self.observed_score_state + self.observed_subgrad,
                             solve_args=solve_args,
                             level=level,
                             useC=self.useC)

    def _log_density_ray(self,
                         candidate,
                         direction,
                         nuisance,
                         gaussian_sample,
                         opt_sample):

        # implicitly caching (opt_sample, gaussian_sample) ?

        if (not hasattr(self, "_direction") or not
        np.all(self._direction == direction)):

            regress_opt, subgrad = self.regress_opt, self.observed_subgrad

            if opt_sample.shape[1] == 1:

                prec = 1. / self.covariance[0, 0]
                quadratic_term = regress_opt.dot(direction) ** 2 * prec
                arg = (opt_sample[:, 0] -
                       regress_opt.dot(nuisance + subgrad) -
                       regress_opt.dot(direction) * gaussian_sample) 
                linear_term = -regress_opt.dot(direction) * prec * arg
                constant_term = arg ** 2 * prec

                self._cache = {'linear_term': linear_term,
                               'quadratic_term': quadratic_term,
                               'constant_term': constant_term}
            else:
                self._direction = direction.copy()

                # density is a Gaussian evaluated at
                # O_i - A(N + (Z_i + theta) * gamma + u)

                # u is observed_subgrad
                # A is regress_opt
                # Z_i is gaussian_sample[i] (real-valued)
                # gamma is direction
                # O_i is opt_sample[i]

                # let arg1 = O_i
                # let arg2 = A(N+u + Z_i \cdot gamma)
                # then it is of the form (arg1 - arg2 - theta * A gamma)

                regress_opt, subgrad = self.regress_opt, self.observed_subgrad
                cov = self.covariance
                prec = np.linalg.inv(cov)
                linear_part = -regress_opt.dot(direction)  # -A gamma

                if 1 in opt_sample.shape:
                    pass  # stop3 what's this for?
                cov = self.covariance

                quadratic_term = linear_part.T.dot(prec).dot(linear_part)

                arg1 = opt_sample.T
                arg2 = -regress_opt.dot(np.multiply.outer(direction, gaussian_sample) +
                                        (nuisance + subgrad)[:, None])
                arg = arg1 + arg2
                linear_term = -regress_opt.T.dot(prec).dot(arg)
                constant_term = np.sum(prec.dot(arg) * arg, 0)

                self._cache = {'linear_term': linear_term,
                               'quadratic_term': quadratic_term,
                               'constant_term': constant_term}
        (linear_term,
         quadratic_term,
         constant_term) = (self._cache['linear_term'],
                           self._cache['quadratic_term'],
                           self._cache['constant_term'])
        return (-0.5 * candidate ** 2 * quadratic_term -
                candidate * linear_term - 0.5 * constant_term)


class optimization_intervals(object):

    def __init__(self,
                 opt_sampling_info,  # a sequence of
                 # (opt_sampler,
                 #  opt_sample,
                 #  opt_logweights,
                 #  cov_target,
                 #  score_cov) objects
                 #  in theory all cov_target
                 #  should be about the same...
                 observed,
                 nsample,  # how large a normal sample
                 cov_target=None,
                 normal_sample=None):

        # not all opt_samples will be of the same size as nsample
        # let's repeat them as necessary

        tiled_sampling_info = []
        for (opt_sampler,
             opt_sample,
             opt_logW,
             t_cov,
             t_score_cov) in opt_sampling_info:
            if opt_sample is not None:
                if opt_sample.shape[0] < nsample:
                    if opt_sample.ndim == 1:
                        tiled_opt_sample = np.tile(opt_sample,
                                                   int(np.ceil(nsample /
                                                               opt_sample.shape[0])))[:nsample]
                        tiled_opt_logW = np.tile(opt_logW,
                                                 int(np.ceil(nsample /
                                                             opt_logW.shape[0])))[:nsample]
                    else:
                        tiled_opt_sample = np.tile(opt_sample,
                                                   (int(np.ceil(nsample /
                                                                opt_sample.shape[0])), 1))[:nsample]
                        tiled_opt_logW = np.tile(opt_logW,
                                                 (int(np.ceil(nsample /
                                                              opt_logW.shape[0])), 1))[:nsample]
                else:
                    tiled_opt_sample = opt_sample[:nsample]
                    tiled_opt_logW = opt_logW[:nsample]
            else:
                tiled_sample = None
            tiled_sampling_info.append((opt_sampler,
                                        tiled_opt_sample,
                                        tiled_opt_logW,
                                        t_cov,
                                        t_score_cov))

        self.opt_sampling_info = tiled_sampling_info
        self._logden = 0
        for opt_sampler, opt_sample, opt_logW, _, _ in opt_sampling_info:

            self._logden += opt_sampler.log_cond_density(
                opt_sample,
                opt_sampler.observed_score_state,
                transform=None)
            self._logden -= opt_logW
            if opt_sample.shape[0] < nsample:
                self._logden = np.tile(self._logden,
                                       int(np.ceil(nsample /
                                                   opt_sample.shape[0])))[:nsample]

        # this is our observed unpenalized estimator
        self.observed = observed.copy()

        # average covariances in case they might be different

        if cov_target is None:
            self.cov_target = 0
            for _, _, _, cov_target, _ in opt_sampling_info:
                self.cov_target += cov_target
            self.cov_target /= len(opt_sampling_info)

        if normal_sample is None:
            self._normal_sample = np.random.multivariate_normal(
                mean=np.zeros(self.cov_target.shape[0]),
                cov=self.cov_target,
                size=(nsample,))
        else:
            self._normal_sample = normal_sample

    def pivot(self,
              linear_func,
              candidate,
              alternative='twosided'):
        '''
        alternative : ['greater', 'less', 'twosided']
            What alternative to use.
        Returns
        -------
        pvalue : np.float
        '''

        if alternative not in ['greater', 'less', 'twosided']:
            raise ValueError("alternative should be one of ['greater', 'less', 'twosided']")

        observed_stat = self.observed.dot(linear_func)
        sample_stat = self._normal_sample.dot(linear_func)

        cov_target = linear_func.dot(self.cov_target.dot(linear_func))

        nuisance = []
        translate_dirs = []

        for (opt_sampler,
             opt_sample,
             _,
             _,
             regress_target_score) in self.opt_sampling_info:
            cur_score_cov = linear_func.dot(regress_target_score)

            # cur_nuisance is in the view's score coordinates
            cur_nuisance = opt_sampler.observed_score_state - cur_score_cov * observed_stat / cov_target
            nuisance.append(cur_nuisance)
            translate_dirs.append(cur_score_cov / cov_target)

        weights = self._weights(sample_stat,  # normal sample
                                candidate,  # candidate value
                                nuisance,  # nuisance sufficient stats for each view
                                translate_dirs)  # points will be moved like sample * regress_target_score

        pivot = np.mean((sample_stat + candidate <= observed_stat) * weights) / np.mean(weights)

        if alternative == 'twosided':
            return 2 * min(pivot, 1 - pivot)
        elif alternative == 'less':
            return pivot
        else:
            return 1 - pivot

    def confidence_interval(self,
                            linear_func,
                            level=0.90,
                            how_many_sd=20,
                            guess=None):

        sample_stat = self._normal_sample.dot(linear_func)
        observed_stat = self.observed.dot(linear_func)

        def _rootU(gamma):
            return self.pivot(linear_func,
                              observed_stat + gamma,
                              alternative='less') - (1 - level) / 2.

        def _rootL(gamma):
            return self.pivot(linear_func,
                              observed_stat + gamma,
                              alternative='less') - (1 + level) / 2.

        if guess is None:
            grid_min, grid_max = -how_many_sd * np.std(sample_stat), how_many_sd * np.std(sample_stat)
            upper = bisect(_rootU, grid_min, grid_max)
            lower = bisect(_rootL, grid_min, grid_max)

        else:
            delta = 0.5 * (guess[1] - guess[0])

            # find interval bracketing upper solution
            count = 0
            while True:
                Lu, Uu = guess[1] - delta, guess[1] + delta
                valU = _rootU(Uu)
                valL = _rootU(Lu)
                if valU * valL < 0:
                    break
                delta *= 2
                count += 1
            upper = bisect(_rootU, Lu, Uu)

            # find interval bracketing lower solution
            count = 0
            while True:
                Ll, Ul = guess[0] - delta, guess[0] + delta
                valU = _rootL(Ul)
                valL = _rootL(Ll)
                if valU * valL < 0:
                    break
                delta *= 2
                count += 1
            lower = bisect(_rootL, Ll, Ul)
        return lower + observed_stat, upper + observed_stat

    # Private methods

    def _weights(self,
                 stat_sample,
                 candidate,
                 nuisance,
                 translate_dirs):

        # Here we should loop through the views
        # and move the score of each view
        # for each projected (through linear_func) normal sample
        # using the linear decomposition

        # We need access to the map that takes observed_score for each view
        # and constructs the full randomization -- this is the reconstruction map
        # for each view

        # The data state for each view will be set to be N_i + A_i \hat{\theta}_i
        # where N_i is the nuisance sufficient stat for the i-th view's
        # data with respect to \hat{\theta} and N_i  will not change because
        # it depends on the observed \hat{\theta} and observed score of i-th view

        # In this function, \hat{\theta}_i will change with the Monte Carlo sample

        score_sample = []
        _lognum = 0
        for i, opt_info in enumerate(self.opt_sampling_info):
            opt_sampler, opt_sample = opt_info[:2]

            _lognum += opt_sampler.log_cond_density(opt_sample,
                                                    stat_sample + candidate,
                                                    transform=
                                                    (translate_dirs[i],
                                                     nuisance[i]))

        _logratio = _lognum - self._logden
        _logratio -= _logratio.max()

        return np.exp(_logratio)


def naive_confidence_intervals(diag_cov, observed, level=0.9):
    """
    Compute naive Gaussian based confidence
    intervals for target.
    Parameters
    ----------
    diag_cov : diagonal of a covariance matrix
    observed : np.float
        A vector of observed data of shape `target.shape`
    alpha : float (optional)
        1 - confidence level.
    Returns
    -------
    intervals : np.float
        Gaussian based confidence intervals.
    """
    alpha = 1 - level
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    quantile = - ndist.ppf(alpha / 2)
    LU = np.zeros((2, p))
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        LU[0, j] = observed[j] - sigma * quantile
        LU[1, j] = observed[j] + sigma * quantile
    return LU.T


def naive_pvalues(diag_cov, observed, parameter):
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    pvalues = np.zeros(p)
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        pval = ndist.cdf((observed[j] - parameter[j]) / sigma)
        pvalues[j] = 2 * min(pval, 1 - pval)
    return pvalues

def selective_MLE(target_spec,
                  observed_soln,  # initial (observed) value of
                  # optimization variables -- used as a
                  # feasible point.  precise value used
                  # only for independent estimator
                  cond_mean,
                  cond_cov,
                  linear_part,
                  offset,
                  opt_linear,
                  M1,   
                  M2,
                  M3,
                  observed_score,  # LASSO: observed_score_state + observed_subgrad
                  solve_args={'tol': 1.e-12},
                  level=0.9,
                  useC=False):

    """
    Selective MLE based on approximation of
    CGF.
    Parameters
    ----------
    observed_target : ndarray
        Observed estimate of target.
    cov_target : ndarray
        Estimated covaraince of target.
    regress_target_score : ndarray
        Estimated regression coefficient of target on score.
    observed_soln : ndarray
        Feasible point for optimization problem.
    cond_mean : ndarray
        Conditional mean of optimization variables given target.
    cond_cov : ndarray
        Conditional covariance of optimization variables given target.
    regress_opt : ndarray
        Describes how conditional mean of optimization
        variables varies with target.
    linear_part : ndarray
        Linear part of affine constraints: $\{o:Ao \leq b\}$
    offset : ndarray
        Offset part of affine constraints: $\{o:Ao \leq b\}$
    solve_args : dict, optional
        Arguments passed to solver.
    level : float, optional
        Confidence level.
    useC : bool, optional
        Use python or C solver.
    """

    #### For LASSO
    # observed_score_state = -X^TY
    # cond_mean = A beta_hat_s + b = -(Sigma_bar Q_s' Sigma_w^{-1}) (P_s beta_hat_s + r_s),
    # where P_s = -X'X_E,
    # and   r_s = 'observed_subgrad' + beta_s^perp
    # observed_score = observed_score_state + observed_subgrad = -X'Y + subgrad
    # cond_precision = Sigma_bar^{-1}
    # cond_cov = Sigma_bar
    # regress_opt = -(Sigma_bar Q_s' Sigma_w^{-1})
    # M1 = dispersion * X'X Sigma_w^{-1}
    # M2 = dispersion^2 * X'X Sigma_w^{-1} X'X
    # M3 = dispersion^2 * X'X Sigma_w^{-1} (Q_s Sigma_bar Q_s') Sigma_w^{-1} X'X

    (observed_target,                           # OLS solution: \hat\beta_S = (X_E'X_E)^-1 X_E Y
     cov_target,                                # dispersion * (X_E'X_E)^-1
     regress_target_score) = target_spec[:3]    # [ (X_E'X_E)^-1  0_{-E} ]

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)     # dispersion^-1 * X_E'X_E

    prec_opt = np.linalg.inv(cond_cov)          # Sigma_bar

    # this is specific to target

    # T1 = dispersion^-1 * [   I_E  ]
    #                      [ 0_{-E} ]
    T1 = regress_target_score.T.dot(prec_target)
    # T2 = X_E'X Sigma_w^{-1} X'X_E
    T2 = T1.T.dot(M2.dot(T1))
    # T3 = X_E'X Sigma_w^{-1} (Q_s Sigma_bar Q_s') Sigma_w^{-1} X'X_E
    T3 = T1.T.dot(M3.dot(T1))
    # For LASSO, opt_linear = Q_s
    # T4 = dispersion * X'X Sigma_w^{-1} (Q_s Sigma_bar Q_s') Sigma_w^{-1} X'X_E
    T4 = M1.dot(opt_linear).dot(cond_cov).dot(opt_linear.T.dot(M1.T.dot(T1)))
    # T5 = X_E'X Sigma_w^{-1} Q_s
    T5 = T1.T.dot(M1.dot(opt_linear))

    # prec_target_nosel = Sigma^-1
    prec_target_nosel = prec_target + T2 - T3

    # _P = - (X_E'X Sigma_w^{-1} (-X'Y + subgrad) + X_E'X Sigma_w^{-1} X'X_E beta_hat_s)
    # _P = - X_E'X Sigma_w^{-1} r_s
    # _P = P_s' Sigma_w^{-1} r_s
    _P = -(T1.T.dot(M1.dot(observed_score)) + T2.dot(observed_target)) ##flipped sign of second term here

    ####################COMMENTS ABOVE SHOULD BE CORRECT#############

    # bias_target = Sigma_{Ms_s} \
    #              @  { [ (P_s' Sigma_w^{-1} (Q_s Sigma_bar Q_s') Sigma_w^{-1} P_s beta_hat_s)
    #                    - P_s' Sigma_w^{-1} Q_s (A beta_hat_s + b) ]
    #                    - P_s' Sigma_w^{-1} r_s }
    bias_target = cov_target.dot(T1.T.dot(-T4.dot(observed_target) + M1.dot(opt_linear.dot(cond_mean))) - _P)

    # conjugate_arg = Sigma_bar (A beta_hat_s + b)
    conjugate_arg = prec_opt.dot(cond_mean)

    if useC:
        solver = solve_barrier_affine_C
    else:
        solver = solve_barrier_affine_py

    # Solution for o_1^*(beta_hat_s)
    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             observed_soln,
                             linear_part,
                             offset,
                             **solve_args)

    # final_estimator = selective_MLE
    # 1. (CORRECT)
    #    cov_target.dot(prec_target_nosel).dot(observed_target)
    #    == J^{-1} beta_hat_s
    # 2. (CORRECT)
    #    regress_target_score.dot(M1.dot(opt_linear)).dot(cond_mean - soln)
    #    == Sigma_{Ms_s} A' Sigma_bar^{-1} (A beta + b - o_1^*(beta_hat_s))
    # 3. (CORRECT)
    #    bias_target = J^{-1} k
    final_estimator = cov_target.dot(prec_target_nosel).dot(observed_target) \
                      + regress_target_score.dot(M1.dot(opt_linear)).dot(cond_mean - soln) - bias_target

    # middle of observed info
    observed_info_natural = prec_target_nosel + T3 - T5.dot(hess.dot(T5.T))

    unbiased_estimator = cov_target.dot(prec_target_nosel).dot(observed_target) - bias_target

    # observed info
    observed_info_mean = cov_target.dot(observed_info_natural.dot(cov_target))

    # individual z-scores
    Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

    pvalues = ndist.cdf(Z_scores)

    pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

    alpha = 1. - level

    quantile = ndist.ppf(1 - alpha / 2.)

    intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                           final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

    log_ref = val + conjugate_arg.T.dot(cond_cov).dot(conjugate_arg) / 2.

    result = pd.DataFrame({'MLE': final_estimator,
                           'SE': np.sqrt(np.diag(observed_info_mean)),
                           'Zvalue': Z_scores,
                           'pvalue': pvalues,
                           'lower_confidence': intervals[:, 0],
                           'upper_confidence': intervals[:, 1],
                           'unbiased': unbiased_estimator})

    return result, observed_info_mean, log_ref

