from __future__ import print_function
import functools
from copy import copy

import numpy as np
from scipy.stats import norm as ndist

import regreg.api as rr

from .query import query, affine_gaussian_sampler

from .randomization import randomization
from .approx_reference_grouplasso import group_lasso
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
         self.perturb = perturb  # random perturbation

         self.randomizer_scale = randomizer_scale

    # TESTED
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

    # TESTED
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

    # TESTED
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
        groups = list(map(int, groups))

        return groups, groups_to_vars, group_weights

    # TESTED
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

    # TESTED
    # Cast the vectorized parameters to a matrix with zero diagonals
    def vec_to_mat(self, p, vec):
        mat = np.zeros((p, p))
        for k in range(len(vec)):
            i,j = self.undo_vectorize(k)
            # print(k, 'mapped to', i, j)
            mat[i,j] = vec[k]
        return mat

    # TESTED
    # Given an index pair (i,j) of the B matrix,
    # this function returns the corresponding index in the vectorized parameter
    def mat_idx_to_vec_idx(self, i, j, p):
        if i < j:
            return (p-1)*j + i
        return (p-1)*j + i - 1

    # TESTED
    # The inverse of vec_to_mat()
    # This is the vectorization operator
    def mat_to_vec(self, p, mat):
        vec = np.zeros((p*(p-1),))
        for i in range(p):
            for j in range(p):
                if i != j:
                    vec_idx = self.mat_idx_to_vec_idx(i,j,p)
                    vec[vec_idx] = mat[i,j]
        return vec

    # REQUIRES: perturb is a p x p ndarray with the diagonal being zero
    def fit(self):
        ### SELECTION PART
        # Vectorize the perturbation
        if self.perturb is not None:
            perturb_vec = self.mat_to_vec(p=self.nfeature, mat=self.perturb)
            #print(self.X)
            #print(self.X_aug)
            glsolver = group_lasso.gaussian(self.X_aug,
                                            self.Y_aug,
                                            np.array(self.groups),
                                            self.weights,
                                            randomizer_scale=self.randomizer_scale,
                                            perturb=perturb_vec)
            signs = glsolver.fit()
        else:
            glsolver = group_lasso.gaussian(self.X_aug,
                                            self.Y_aug,
                                            np.array(self.groups),
                                            self.weights,
                                            randomizer_scale=self.randomizer_scale)
            signs = glsolver.fit()
            # If perturbation not provided, stack the perturbation given by glslover into matrix
            perturb_vec = glsolver._initial_omega
            self.perturb = self.vec_to_mat(p=self.nfeature, vec=perturb_vec)

        # coeffs = signs['directions']
        # nonzero = glsolver.selection_variable['directions'].keys()

        # gammas negative in original implementation?
        gammas = glsolver.observed_opt_state
        # subgrad is the subgradient vector corresponding to beta,
        # with each of its entries scaled up by the correpsponding penalty weight
        subgrad = glsolver.initial_subgrad

        # Cast the subgradient vector into matrix
        subgrad_mat = self.vec_to_mat(p=self.nfeature, vec=subgrad)
        #print(subgrad_mat)
        #print('gamma',gammas)
        #print('subgrad',subgrad)

        vectorized_beta = glsolver.initial_soln
        # Stack the parameters into a pxp matrix
        beta = self.vec_to_mat(p = self.nfeature, vec=vectorized_beta)
        #print('beta_vec', vectorized_beta)
        #print('beta', beta)

        # KKT map for the group lasso solver
        # LHS_gl = glsolver._initial_omega
        # RHS_gl = -(self.X_aug.T @ self.Y_aug) + self.X_aug.T @ self.X_aug @ vectorized_beta + subgrad
        # num_disagreement = np.abs(LHS_gl - RHS_gl) > 0.00000001
        # print(num_disagreement)
        # num_dis_mat = self.vec_to_mat(p = self.nfeature, vec=num_disagreement.astype(int)).astype(bool)
        # print('gl disagreement', num_dis_mat)

        print(self.X_aug.T @ self.X_aug @ vectorized_beta)

        # Calculate the KKT map for the paired group lasso
        rhs = - self.X.T @ self.X + (self.X.T @ self.X) @ beta + subgrad_mat
        np.fill_diagonal(rhs, 0)
        lhs = self.perturb

        # print('paired gl disagreement', np.abs(lhs - rhs) > 0.00000001)
        # print('lhs', lhs)
        # print('rhs', rhs)
        # print('rhs_iter', rhs_iter)
        self.observed_opt_state = glsolver.observed_opt_state
        self.opt_linear = glsolver.opt_linear
        self.observed_score_state = glsolver.observed_score_state
        self.opt_offset = glsolver.opt_offset
        self._initial_omega = glsolver._initial_omega
        self.ordered_groups = glsolver._ordered_groups
        #print("sub", glsolver.initial_subgrad)
        #print('vars', glsolver.ordered_vars)

        # -(self.X_aug.T @ self.Y_aug) == pgl.observed_score_state
        # print(np.abs(-(self.X_aug.T @ self.Y_aug) - self.observed_score_state))

        # self.opt_linear.dot(self.observed_opt_state) = X^T sum(X gamma u)
        # roughly the same
        # print(np.abs(self.opt_linear.dot(self.observed_opt_state) - (self.X_aug.T @ self.X_aug) @ vectorized_beta))

        self.beta = beta

        ### INFERENCE PART
        ## X_ is the augmented design matrix
        ## Y_ is the augmented response
        ## t is the value of x_i^T x_j
        ## REQUIRES: i < j, i, j follows the natural enumeration (starting from 1),
        ##           p is the dimension of data,
        def XY(t, i, j, p, X_, Y_):
            i = i - 1
            j = j - 1

            # the first appearance of x_i^T x_j
            idx_ij = i*(p-1) + j - 1
            idx_ji = j*(p-1) + i
            # the target object
            XY = X_.T @ Y_
            XY[idx_ij] = t
            XY[idx_ji] = t
            return XY

        ## NOTES: Assuming i < j, then b_ij comes after b_ji in the vectorized beta
        ##        This implies when we order covariates according to groups,
        ##        within the group g corresponding to i,j,
        ##        the earlier one corresponds to b_ji, and the later one corresponds to b_ij.
        ##        That is, the earlier column contains the observations x_j,
        ##        and the later column contains the observations x_i.
        ## REQUIRES: i < j, i, j follows the natural enumeration (starting from 1),
        ##           p is the dimension of data, g is the group label of i,j,
        ##           Retrieval of g: g = groups[j * (p - 1) + i]
        def XXE(t, i, j, p, X_, XE):
            i = i - 1
            j = j - 1

            # the row index corresponding to x_j^T x_i
            row_ji = i*(p-1) + j - 1
            # the row index corresponding to x_i^T x_j
            row_ij = j*(p-1) + i

            # g is the group label of i,j
            g = self.groups[j * (p - 1) + i]
            # g_i is the index of g in the list of ordered selected groups
            g_i = self.ordered_groups.index(g)
            # the column index corresponding to x_j^T x_i
            col_ji = 2*g_i + 1        # all groups are of size 2
            # the column index corresponding to x_i^T x_j
            col_ij = 2*g_i

            # the target object
            XXE_ = X_.T @ XE
            XXE_[row_ji, col_ji] = t
            XXE_[row_ij, col_ij] = t
            return XXE_

        ## NOTES: beta_grouped is the solution of beta ordered according to groups
        ##        Retrieval of beta_grouped: beta_grouped = initial_soln[ordered_vars]
        ##        prec:
        def quad_exp(t, X_, Y_, XE, p, prec,
                     beta_grouped, subgradient, i, j):
            XY_ = XY(t=t, i=i, j=j, p=p, X_=X_, Y_=Y_)
            XXE_ = XXE(t=t, i=i, j=j, p=p, X_=X_, XE=XE)
            omega = -XY_ + XXE_ @ beta_grouped + subgradient
            return np.exp(omega.T @ prec @ omega)

        def call_quad_exp(i,j):
            t = self.X[:,i-1].T @ self.X[:,j-1]
            X_ = self.X_aug
            XE = glsolver.XE
            Y_ = self.Y_aug
            p = self.nfeature
            beta_grouped = glsolver.initial_soln[glsolver.ordered_vars]
            subgradient = glsolver.initial_subgrad
            quad_t = quad_exp(t=t, X_=X_, Y_=Y_, XE=XE, p=p, prec=np.identity(p*(p-1)),
                              beta_grouped=beta_grouped, subgradient=subgradient, i=i, j=j)
            quad_original = np.exp(glsolver._initial_omega.T @ glsolver._initial_omega)

            #print(quad_t)
            #print(quad_original)
            #print(glsolver._initial_omega)
            #print(XXE_t)
            #print(glsolver._initial_omega)

            #T1
            #print(-X_.T @ Y_)
            #T2
            print(X_.T @ XE @ beta_grouped)

        call_quad_exp(2,4)
