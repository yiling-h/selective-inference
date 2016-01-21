"""
This module contains a class `logistic`_ that implements
post selection inference for logistic LASSO.

"""

import warnings
from copy import copy

import numpy as np

from regreg.api import weighted_l1norm, simple_problem
from regreg.smooth.glm import coxph, logistic, gaussian, poisson

from ..constraints.affine import constraints

class OLS_lasso(object):

    r"""
    A class for the LASSO for post-selection inference.
    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta;X,Y) + 
             \sum_{i=1}^p \lambda_i \|\beta\|_i

    where $\lambda$ is `feature_weights` and $\ell$ is 
    $1/2 SSE(\beta)$, the negative Gaussian log-likelihood.

    """

    def __init__(self, X, Y, feature_weights):
        r"""

        Usual LASSO problem.

        Parameters
        ----------

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        Y : np.float(n)
            Event times

        E : np.bool(n)
            Censoring variable.

        feature_weights : np.float(p)
            Feature weights for L-1 penalty in lagrange form.

        """
        self.X, self.Y = X, Y
        self.loss = gaussian(X, Y)
        self.feature_weights = np.asarray(feature_weights)
        if not self.feature_weights.shape:
            self.feature_weights = np.ones(self.X.shape[1]) * self.feature_weights
        self.unpenalized = np.nonzero((self.feature_weights == 0))[0]
        self.penalized = np.nonzero((self.feature_weights != 0))[0]

    def fit(self, solve_args={'min_its':50, 'tol':1.e-10}):
        """
        Fit the lasso using `regreg`.
        This sets the attribute `soln` and
        forms the constraints necessary for post-selection inference
        by calling `form_constraints()`.

        Parameters
        ----------

        solve_args : {}
             Passed to `regreg.simple_problem.solve``_

        Returns
        -------

        """

        n, p = self.loss.X.shape
        loss = self.loss
        penalty = weighted_l1norm(self.feature_weights, lagrange=1.)
        problem = simple_problem(loss, penalty)
        soln = problem.solve(**solve_args)

        self._soln = soln
        if not np.all(soln[self.penalized] == 0):
            self.active = np.nonzero(soln)[0]
            self.signs = np.sign(soln[self.active])
            self.subgrad = -self.loss.smooth_objective(soln, 'grad')[self.active]
            self.Q = self.loss.hessian(soln)[:,self.active][self.active]
            step = np.linalg.solve(self.Q, self.subgrad)
            self._onestep = soln[self.active] + step

            # ignore unpenalized coordiantes

            _active_pen = np.array([j not in self.unpenalized for
                                        j in self.active])
            _active_selector = np.identity(self.active.shape[0])[_active_pen]

            self._active_constraints = constraints(-self.signs[_active_pen,None] * 
                                                    _active_selector,
                                                    -(self.signs * step)[_active_pen],
                                                    covariance=np.linalg.inv(
                                                    self.Q))
        else:
            self.active = []
            self._active_constraints = None

    @property
    def soln(self):
        """
        Solution to the lasso problem, set by `fit` method.
        """
        if not hasattr(self, "_soln"):
            self.fit()
        return self._soln

    @property
    def active_constraints(self):
        """
        Affine constraints imposed on the
        active variables by the KKT conditions.
        """
        return self._active_constraints

    @property
    def active_pvalues(self, doc="Tests for active variables adjusted " + \
        " for selection."):
        if not hasattr(self, "_pvals"):
            self._pvals = []
            if len(self.active) > 0:
                C = self.active_constraints
                I = np.identity(self.active.shape[0])
                for j in range(self.active.shape[0]):
                    eta = I[j]
                    _pval = C.pivot(eta, self._onestep)
                    _pval = 2 * min(_pval, 1 - _pval)
                    self._pvals.append((self.active[j], _pval))
        return self._pvals

    @property
    def intervals(self):
        """
        Intervals for OLS parameters of active variables
        adjusted for selection.

        """
        if not hasattr(self, "_intervals"):
            self._intervals = []
            C = self.active_constraints
            I = np.identity(self.active.shape[0])
            for j in range(self.active.shape[0]):
                eta = I[j]
                _interval = C.interval(eta, self._onestep,
                                       alpha=self.alpha)
                self._intervals.append((self.active[i],
                                        _interval[0], _interval[1]))
            self._intervals = np.array(self._intervals, 
                                       np.dtype([('index', np.int),
                                                 ('lower', np.float),
                                                 ('upper', np.float)]))
        return self._intervals

class PH_lasso(OLS_lasso):

    r"""
    A class for the LASSO for post-selection inference.

    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta;X,Y) + 
             \sum_{i=1}^p \lambda_i \|\beta\|_i

    where $\lambda$ is `feature_weights` and $\ell$ is 
    (negative) Cox partial log-likelihood.

    """

    def __init__(self, X, T, E, feature_weights):
        r"""

        Create a new post-selection dor the LASSO problem

        Parameters
        ----------

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        T : np.float(n)
            Event times

        E : np.bool(n)
            Censoring variable.

        feature_weights : np.float(p)
            Feature weights for L-1 penalty in lagrange form.

        """
        self.X, self.T, self.E = X, T, E
        self.loss = coxph(X, T, E)
        self.feature_weights = np.asarray(feature_weights)
        if not self.feature_weights.shape:
            self.feature_weights = np.ones(self.X.shape[1]) * self.feature_weights
        self.unpenalized = np.nonzero((self.feature_weights == 0))[0]
        self.penalized = np.nonzero((self.feature_weights != 0))[0]

class Logistic_lasso(OLS_lasso):

    r"""
    A class for the LASSO for post-selection inference.

    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta;X,Y) + 
             \sum_{i=1}^p \lambda_i \|\beta\|_i

    where $\lambda$ is `feature_weights` and $\ell$ is 
    (negative of) the Binomial log-likelihood.

    """

    def __init__(self, X, Y, feature_weights):
        r"""

        Create a new post-selection for the LASSO problem

        Parameters
        ----------

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        Y : np.bool(n)
            Successes.

        feature_weights : np.float
            Coefficient of the L-1 penalty in lagrange form.

        """
        self.X, self.Y = X, Y
        self.loss = logistic(X, Y)
        self.feature_weights = np.asarray(feature_weights)
        if not self.feature_weights.shape:
            self.feature_weights = np.ones(self.X.shape[1]) * self.feature_weights
        self.unpenalized = np.nonzero((self.feature_weights == 0))[0]
        self.penalized = np.nonzero((self.feature_weights != 0))[0]

class Poisson_lasso(OLS_lasso):

    r"""
    A class for the LASSO for post-selection inference.

    The problem solved is

    .. math::

        \text{minimize}_{\beta} \ell(\beta;X,Y) + 
             \sum_{i=1}^p \lambda_i \|\beta\|_i

    where $\lambda$ is `feature_weights` and $\ell$ is 
    (negative of) the Poisson log-likelihood.

    """

    def __init__(self, X, Y, feature_weights):
        r"""

        Create a new post-selection for the LASSO problem

        Parameters
        ----------

        X : np.float((n, p))
            The data, in the model $y = X\beta$

        Y : np.bool(n)
            Successes.

        feature_weights : np.float
            Coefficient of the L-1 penalty in lagrange form.

        """
        self.X, self.Y = X, Y
        self.loss = poisson(X, Y)
        self.feature_weights = np.asarray(feature_weights)
        if not self.feature_weights.shape:
            self.feature_weights = np.ones(self.X.shape[1]) * self.feature_weights
        self.unpenalized = np.nonzero((self.feature_weights == 0))[0]
        self.penalized = np.nonzero((self.feature_weights != 0))[0]

