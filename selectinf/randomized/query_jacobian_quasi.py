from typing import NamedTuple
import numpy as np, pandas as pd

from ..constraints.affine import constraints
from .selective_MLE_jacobian import mle_inference

class QuerySpec(NamedTuple):

    # law of gamma | A_E = {E, U, Z}, \hat{beta_E}

    cond_mean : np.ndarray
    cond_cov : np.ndarray

    # how U enters into E[gamma | A_E, \hat{beta_E}]

    opt_linear : np.ndarray

    # constraints (I \gamma > 0)

    linear_part : np.ndarray
    offset : np.ndarray

    # score / randomization relationship

    M1 : np.ndarray
    M2 : np.ndarray
    M3 : np.ndarray

    # observed values

    observed_opt_state : np.ndarray     # gammas
    observed_score_state : np.ndarray   # -X'Y
    observed_subgrad : np.ndarray       # subgradients scaled by lambda
    observed_soln : np.ndarray          # gammas
    observed_score : np.ndarray         # -X'Y + subgrad = "score_offset"

class JacobianSpec(NamedTuple):

    # Constant term in the Jacobian calculation
    C : np.ndarray

    # Unit-norms representations of the active directions
    active_dirs : dict

    
class gaussian_query(object):
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

    def __init__(self, randomization, useJacobian=False, perturb=None):

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
        self.useJacobian = useJacobian      # logical value for whether a Jacobian is needed
        self.perturb = perturb
        self._solved = False
        self._randomized = False
        self._setup = False

    @property
    def specification(self):
        return QuerySpec(cond_mean=self.cond_mean,
                         cond_cov=self.cond_cov,
                         opt_linear=self.opt_linear,
                         linear_part=self.affine_con.linear_part,
                         offset=self.affine_con.offset,
                         M1=self.M1,
                         M2=self.M2,
                         M3=self.M3,
                         observed_opt_state=self.observed_opt_state,
                         observed_score_state=self.observed_score_state,
                         observed_subgrad=self.observed_subgrad,
                         observed_soln=self.observed_opt_state,
                         observed_score=self.observed_score_state + self.observed_subgrad)

    @property
    def Jacobian_info(self):
        return JacobianSpec(C=self.C,
                            active_dirs=self.active_dirs)

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

    def fit(self, perturb=None):

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

        (cond_mean,
         cond_cov,
         cond_precision,
         M1,
         M2,
         M3) = self._setup_implied_gaussian(opt_linear,
                                            observed_subgrad,
                                            dispersion=dispersion)

        self.cond_mean, self.cond_cov = cond_mean, cond_cov

        affine_con = constraints(A,
                                 b,
                                 mean=cond_mean,
                                 covariance=cond_cov)

        self.affine_con = affine_con
        self.opt_linear = opt_linear
        self.observed_subgrad = observed_subgrad

    def _setup_implied_gaussian(self,
                                opt_linear,
                                observed_subgrad,
                                dispersion=1):

        cov_rand, prec = self.randomizer.cov_prec

        if np.asarray(prec).shape in [(), (0,)]:
            prod_score_prec_unnorm = self._unscaled_cov_score * prec
        else:
            prod_score_prec_unnorm = self._unscaled_cov_score.dot(prec)

        if np.asarray(prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * prec
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T) * prec
        else:
            cond_precision = opt_linear.T.dot(prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            regress_opt = -cond_cov.dot(opt_linear.T).dot(prec)

        # regress_opt is regression coefficient of opt onto score + u...
        cond_mean = regress_opt.dot(self.observed_score_state + observed_subgrad)

        # Remain the same as in LASSO
        M1 = prod_score_prec_unnorm * dispersion
        M2 = M1.dot(cov_rand).dot(M1.T)
        M3 = M1.dot(opt_linear.dot(cond_cov).dot(opt_linear.T)).dot(M1.T)

        self.M1 = M1
        self.M2 = M2
        self.M3 = M3

        return (cond_mean,
                cond_cov,
                cond_precision,
                M1,
                M2,
                M3)

    def inference(self,
                  target_spec,
                  method,
                  level=0.90,
                  method_args={}):

        """
        Parameters
        ----------
        target_spec : TargetSpec
           Information needed to specify the target.
        method : str
           One of ['selective_MLE', 'approx', 'exact', 'posterior']
        level : float
           Confidence level or posterior quantiles.
        method_args : dict
           Dict of arguments to be optionally passed to the methods.

        Returns
        -------

        summary : pd.DataFrame
           Statistical summary for specified targets.
        """

        query_spec = self.specification

        if not hasattr(self, "useJacobian"):
            self.useJacobian = False

        if self.useJacobian:
            Jacobian_spec = self.Jacobian_info

        if method == 'selective_MLE':
            if self.useJacobian:
                G = mle_inference(query_spec,
                                  target_spec,
                                  self.useJacobian,
                                  Jacobian_spec,
                                  **method_args)
            else:
                G = mle_inference(query_spec,
                                  target_spec,
                                  self.useJacobian,
                                  None,
                                  **method_args)

            return G.solve_estimating_eqn(alternatives=target_spec.alternatives,
                                          level=level)[0]







