from __future__ import print_function
import numpy as np
from scipy.linalg import block_diag

import regreg.api as rr

from .query import query, affine_gaussian_sampler

from .approx_reference_grouplasso import group_lasso
class paired_group_lasso(group_lasso):
    def  __init__(self,
                  loglike,
                  weights,
                  ridge_term,
                  randomizer_scale,
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
         # log likelihood : quadratic loss
         self.loglike = loglike
         # ridge parameter
         self.ridge_term = ridge_term
         # perturbation
         self.perturb = perturb  # random perturbation
         # gaussian randomization
         self.randomizer = randomizer



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

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50},
            perturb=None):
        ### SELECTION PART

    @staticmethod
    def Gaussian(self,
                 X,
                 weights,
                 ridge_term,
                 randomizer_scale,
                 perturb=None):
        self.X = X
        self.nfeature = p = self.X.shape[1]
        self.nobs = n = self.X.shape[0]
        self.X_aug = self.augment_X()
        self.Y_aug = self.augment_Y()

        self.groups, self.groups_to_vars, self.weights = self.create_groups(weights)

        # Optimization hyperparameters
        self.penalty = rr.group_lasso(self.groups,
                                      weights=self.weights,
                                      lagrange=1.)

        self.randomizer_scale = randomizer_scale
