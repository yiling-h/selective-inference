from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from .query import query, affine_gaussian_sampler

from .randomization import randomization
from .group_lasso import group_lasso
from ..base import restricted_estimator
from ..algorithms.debiased_lasso import (debiasing_matrix,
                                         pseudoinverse_debiasing_matrix)

class paired_group_lasso(query):
    def  __init__(self,
                  X,
                  weights,
                  ridge_term,
                  randomizer,
                  perturb=None):
         r"""
         Create a new post-selection object for the paired group LASSO problem

         Parameters
         ----------

         weights : np.ndarray
             Feature weights for L-1 penalty. If a float,
             it is broadcast to all features.

         ridge_term : float
             How big a ridge term to add?

         randomizer : object
             Randomizer -- contains representation of randomization density.

         perturb : np.ndarray
             Random perturbation subtracted as a linear
             term in the objective function.
         """

         self.X = X
         self.nfeature = p = self.X.shape[1]

         self.ridge_term = ridge_term
         self.penalty = rr.group_lasso(groups,
                                       weights=weights,
                                       lagrange=1.)
         self._initial_omega = perturb  # random perturbation

         self.randomizer = randomizer

    def augment_X(self):
        r"""
        Augment the matrix X to get a design matrix used for the group lasso solver.
        """
        n = self.X.shape[0]
        p = self.X.shape[1]
        q = p - 1

        X_aug = np.zeros((p*n, q*p))
        for j in range(p):
            X_aug[(j*n):(j*n+n),(j*q):(j*q+q)] = np.delete(self.X, j, axis=1)

        return X_aug

    def augment_Y(self):
        r"""
        Generate an augmented vector Y to get a response vector used for the group lasso solver.
        """
        n = self.X.shape[0]
        p = self.X.shape[1]

        Y_aug = np.zeros((p*n,))
        for j in range(p):
            Y_aug[(j*n):(j*n+n),] = self.X[:,j]

        return Y_aug

    def create_groups(self):
        r"""
        Generate an ndarray containing the appropriate grouping of parameters: (b_ij, b_ji)
        """
        n = self.X.shape[0]
        p = self.X.shape[1]

        groups = np.zeros((p*(p-1),))

        for i in range(p):
            for j in range(i+1,p):


        return groups
