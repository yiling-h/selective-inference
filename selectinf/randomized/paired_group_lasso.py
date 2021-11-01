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
                  randomizer_scale,
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
         self.nobs = n = self.X.shape[0]
         self.X_aug = self.augment_X()
         self.Y_aug = self.augment_Y()

         self.groups, self.groups_to_vars, self.weights = self.create_groups(weights)

         # Optimization hyperparameters
         self.ridge_term = ridge_term
         self.penalty = rr.group_lasso(self.groups,
                                       weights=self.weights,
                                       lagrange=1.)
         self._initial_omega = perturb  # random perturbation

         self.randomizer_scale = randomizer_scale

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

    def create_groups(self, weights):
        r"""
        1. Generate an ndarray containing the appropriate grouping of parameters: (b_ij, b_ji)
        2. Generate a dict that maps back from g to (i,j), i<j
        3. Generate a dict for group weights that is interperatable by the group lasso solver,
           from a symmetric ndarray of weights
        """
        n = self.X.shape[0]
        p = self.X.shape[1]

        # E.g. groups = [0, 0, 1, 1, 1]; start counting from 0
        groups = np.zeros((p * (p - 1),))
        g = 0
        groups_to_vars = dict()
        group_weights = dict()

        # indicator of whether weights is a real number
        is_singleton = (type(weights) == float or type(weights) == int)

        for i in range(p):
            for j in range(i + 1, p):
                # Assign b_ij and b_ji to be in the same group
                # Note that b_ji is mapped to an earlier dimension of the vectorized parameter
                groups[j * (p - 1) + i] = g
                groups[i * (p - 1) + j - 1] = g
                # Record this correspondence between g and i,j
                groups_to_vars[g] = [i,j]
                # Record the group weights accordingly
                if is_singleton:
                    group_weights[g] = weights
                else:
                    group_weights[g] = weights[i, j]

                g = g + 1

        # Cast the datatype
        groups = groups.tolist()

        return groups, groups_to_vars, group_weights

    def undo_vectorize(self, k):
        r"""
        1. Mapp the k-th entry of the vectorized parameter to its corresponding
           entry in the matrix parameter
        """
        p = self.X.shape[1]
        j = k // (p-1)
        i = k % (p-1)
        if i >= j:
            i = i + 1

        return i,j

    def vec_to_mat(self, p, vec):
        mat = np.zeros((p, p))
        for k in range(len(vec)):
            i,j = self.undo_vectorize(k)
            # print(k, 'mapped to', i, j)
            mat[i,j] = vec[k]
        return mat

    # REQUIRES: perturb is a p x p ndarray with the diagonal being zero
    def fit(self, perturb=None):
        glsolver = group_lasso.gaussian(self.X_aug,
                                        self.Y_aug,
                                        self.groups,
                                        self.weights,
                                        randomizer_scale=self.randomizer_scale)
        signs = glsolver.fit()
        coeffs = signs['directions']
        nonzero = glsolver.selection_variable['directions'].keys()

        # If perturbation not provided, stack the perturbation given by the glsover into matrix
        if perturb == None:
            perturb_vec = glsolver._initial_omega
            perturb = self.vec_to_mat(p=self.nfeature, vec=perturb_vec)
        self.perturb = perturb

        # gammas negative in original implementation?
        gammas = glsolver.observed_opt_state
        subgrad = glsolver.initial_subgrad
        print('gamma',gammas)
        print('subgrad',subgrad)

        vectorized_beta = glsolver.initial_soln
        # Stack the parameters into a pxp matrix
        beta = self.vec_to_mat(p = self.nfeature, vec=vectorized_beta)
        print('beta_vec', vectorized_beta)
        print(beta)

        """
        # indicator of whether weights is a real number
        is_singleton = (type(self.weights) == float or type(self.weights) == int)
        # the term involving subgradient in the KKT map
        scaled_subgrad = None
        if is_singleton: # when the weights are uniform
            scaled_subgrad = self.weights * subgrad
        else: # assuming subgrad is a valid p x p matrix, self.weights a symmetric matrix
            # Elementwise multiplication that multiplies each subgradient with its weight
            scaled_subgrad = np.multiply(self.weights,subgrad)
        rhs = - self.X @ self.X + (self.X @ self.X) @ beta + scaled_subgrad
        lhs = self.perturb
        """
