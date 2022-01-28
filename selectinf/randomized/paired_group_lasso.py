from __future__ import print_function
import numpy as np
from scipy.linalg import block_diag

import regreg.api as rr
from .randomization import randomization

from .query import query, affine_gaussian_sampler

from .approx_reference_grouplasso import group_lasso
class paired_group_lasso(group_lasso):
    def  __init__(self,
                  loglike,
                  n,p,
                  groups,
                  weights,
                  randomizer,
                  randomizer_scale,
                  ridge_term=0,
                  perturb=None):
        ## What input should this function take?
        ## loglike?
        ## We instantiated Gaussian in fit(), so there is no need to specify loglike

        ## Should we provide alternative versions of fit() so that we instantiate
        ## a group_lasso object with different loglike, instead of Gaussian?

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
         # log-likelihood from augmented data
         self.loglike = loglike
         # NOTE: nfeature = p*(p-1), p is the dimension of the raw data
         self.nfeature = self.loglike.shape[0]
         self.n = n
         self.p = p
         # Augmented matrices
         self.X_aug, self.Y_aug = self.loglike.data

         # perturbation
         self.perturb = perturb  # random perturbation

         self.groups = groups

         self.penalty = rr.group_lasso(groups,
                                       weights=weights,
                                       lagrange=1.)

         self._initial_omega = perturb

         # randomization (typically gaussian)
         self.randomizer = randomizer

         self.ridge_term = ridge_term

    # TESTED
    @staticmethod
    def augment_X(X):
        r"""
        Augment the matrix X to get a design matrix used for the group lasso solver.
        """
        n = X.shape[0]
        p = X.shape[1]
        q = p - 1

        X_aug = np.zeros((p*n, q*p))
        for j in range(p):
            X_aug[(j*n):(j*n+n),(j*q):(j*q+q)] = np.delete(X, j, axis=1)

        return X_aug

    # TESTED
    @staticmethod
    def augment_Y(X):
        r"""
        Generate an augmented vector Y to get a response vector used for the group lasso solver.
        """
        n = X.shape[0]
        p = X.shape[1]

        Y_aug = np.zeros((p*n,))
        for j in range(p):
            Y_aug[(j*n):(j*n+n),] = X[:,j]

        return Y_aug

    # TESTED
    @staticmethod
    def create_groups(n, p, weights):
        r"""
        1. Generate an ndarray containing the appropriate grouping of parameters: (b_ij, b_ji)
        2. Generate a dict that maps back from g to (i,j), i<j
        3. Generate a dict for group weights that is interperatable by the group lasso solver,
           from a symmetric ndarray of weights
        """

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
    @staticmethod
    def undo_vectorize(p, k):
        r"""
        1. Mapp the k-th entry of the vectorized parameter to its corresponding
           entry in the matrix parameter
        """
        j = k // (p-1)
        i = k % (p-1)
        if i >= j:
            i = i + 1

        return i,j

    # TESTED
    # Cast the vectorized parameters to a matrix with zero diagonals
    @staticmethod
    def vec_to_mat(p, vec):
        mat = np.zeros((p, p))
        for k in range(len(vec)):
            i,j = paired_group_lasso.undo_vectorize(p, k)
            mat[i,j] = vec[k]
        return mat

    # TESTED
    # Given an index pair (i,j) of the B matrix,
    # this function returns the corresponding index in the vectorized parameter
    @staticmethod
    def mat_idx_to_vec_idx(i, j, p):
        if i < j:
            return (p-1)*j + i
        return (p-1)*j + i - 1

    # TESTED
    # The inverse of vec_to_mat()
    # This is the vectorization operator
    # p is the number of raw covariates
    @staticmethod
    def mat_to_vec(p, mat):
        vec = np.zeros((p*(p-1),))
        for i in range(p):
            for j in range(p):
                if i != j:
                    vec_idx = paired_group_lasso.mat_idx_to_vec_idx(i,j,p)
                    vec[vec_idx] = mat[i,j]
        return vec

    def fit(self,
            solve_args={'tol': 1.e-12, 'min_its': 50}):
        ### SELECTION PART
        # Vectorize the perturbation
        if self.perturb is not None:
            perturb_vec = self.mat_to_vec(p=self.p, mat=self.perturb)
            active_signs, soln = group_lasso.fit(self, perturb = perturb_vec)
        else:
            active_signs, soln = group_lasso.fit(self, perturb = None)
            perturb_vec = self._initial_omega
            self.perturb = self.vec_to_mat(p=self.p, vec=perturb_vec)
            print(self.perturb)

        # Cast the subgradient vector into matrix
        subgrad_mat = self.vec_to_mat(p=self.p, vec=self.initial_subgrad)

        # Stack the parameters into a pxp matrix
        self.beta = self.vec_to_mat(p=self.p, vec=self.initial_soln)

        ### INFERENCE PART

        ## TESTED
        ## X_ is the augmented design matrix
        ## Y_ is the augmented response
        ## t is the value of x_i^T x_j
        ## REQUIRES: i < j, i, j follows the natural enumeration (starting from 1),
        ##           p is the dimension of data,
        def XY(t, i, j, p, X_, Y_):
            i = i - 1
            j = j - 1

            # the first appearance of x_i^T x_j
            idx_ij = i * (p - 1) + j - 1
            idx_ji = j * (p - 1) + i
            # the target object
            XY = X_.T @ Y_
            XY[idx_ij] = t
            XY[idx_ji] = t
            return XY

        ## TESTED
        ## NOTES: Assuming i < j, then b_ij comes after b_ji in the vectorized beta
        ##        This implies when we order covariates according to groups,
        ##        within the group g corresponding to i,j,
        ##        the earlier one corresponds to b_ji, and the later one corresponds to b_ij.
        ##        That is, the earlier column contains the observations x_j,
        ##        and the later column contains the observations x_i.
        ## REQUIRES: i, j follows the python indexing (starting from 0),
        ##           p is the dimension of data, g is the group label of i,j,
        ##           Retrieval of g: g = groups[j * (p - 1) + i]
        def XXE(t, i, j, p, X_, XE):
            # Swap the value of i,j if i>j,
            # so that i always represents the lower value
            if i > j:
                k = i
                i = j
                j = k

            # the target object
            XXE_ = X_.T @ XE

            # identify x_i*x_j and x_j*x_i in the kth block
            for k in range(p):
                # when both x_i and x_j appear in the kth blcok
                if i != k and j != k:
                    if i > k:
                        i_idx = (p - 1) * k + i - 1
                    else:
                        i_idx = (p - 1) * k + i

                    if j > k:
                        j_idx = (p - 1) * k + j - 1
                    else:
                        j_idx = (p - 1) * k + j

                    # identify x_i^T * x_j if b_jk != 0
                    if self.groups[j_idx] in self._ordered_groups:
                        # g_j is the index of x_j's group
                        # in the list of ordered selected groups
                        g_j = self._ordered_groups.index(self.groups[j_idx])

                        # In our indexing rule that determines the group index
                        # of each parameter, if two augmented vectors, one containing x_i,
                        # one containing x_j, are in the same group, with i < j,
                        # then in the truncated matrix ordered by groups,
                        # the column containing x_j will be the to left of the other,
                        # as explained in the comments above function definition
                        if np.max(self.undo_vectorize(p, j_idx)) == j:
                            j_idx_XE = 2 * g_j
                        else:
                            j_idx_XE = 2 * g_j + 1
                        XXE_[i_idx, j_idx_XE] = t

                    # identify x_j^T * x_i if b_ik != 0
                    if self.groups[i_idx] in self._ordered_groups:
                        # g_i is the index of x_i's group
                        # in the list of ordered selected groups
                        g_i = self._ordered_groups.index(self.groups[i_idx])
                        if np.max(self.undo_vectorize(p, i_idx)) == i:
                            i_idx_XE = 2 * g_i
                        else:
                            i_idx_XE = 2 * g_i + 1
                        XXE_[j_idx, i_idx_XE] = t
            return XXE_

        ## TESTED
        ## NOTES: beta_grouped is the solution of beta ordered according to groups
        ##        Retrieval of beta_grouped: beta_grouped = initial_soln[ordered_vars]
        ##        prec:
        def quad_exp(t, X_, Y_, XE, p, prec,
                     beta_grouped, subgradient, i, j):
            XY_ = XY(t=t, i=i, j=j, p=p, X_=X_, Y_=Y_)
            XXE_ = XXE(t=t, i=i, j=j, p=p, X_=X_, XE=XE)
            omega = -XY_ + XXE_ @ beta_grouped + subgradient
            return np.exp(omega.T @ prec @ omega)

        ## TESTED
        def Q(t, i, j, p, XE):
            # Swap the value of i,j if i>j,
            # so that i always represents the lower value
            if i > j:
                k = i
                i = j
                j = k

            # the target object
            XEXE = XE.T @ XE

            # identify x_i*x_j and x_j*x_i in the kth block
            for k in range(p):
                # when both x_i and x_j appear in the kth blcok
                if i != k and j != k:
                    if i > k:
                        i_idx = (p - 1) * k + i - 1
                    else:
                        i_idx = (p - 1) * k + i

                    if j > k:
                        j_idx = (p - 1) * k + j - 1
                    else:
                        j_idx = (p - 1) * k + j

                    # identify x_i^T * x_j if b_jk != 0 AND b_ik != 0
                    if self.groups[j_idx] in self._ordered_groups and \
                            self.groups[i_idx] in self._ordered_groups:
                        # g_j, g_i is the index of x_j, x_i's group
                        # in the list of ordered selected groups
                        g_j = self._ordered_groups.index(self.groups[j_idx])
                        g_i = self._ordered_groups.index(self.groups[i_idx])

                        # In our indexing rule that determines the group index
                        # of each parameter, if two augmented vectors, one containing x_i,
                        # one containing x_j, are in the same group, with i < j,
                        # then in the truncated matrix ordered by groups,
                        # the column containing x_j will be the to left of the other,
                        # as explained in the comments above function definition
                        if np.max(self.undo_vectorize(p, j_idx)) == j:
                            j_idx_XE = 2 * g_j
                        else:
                            j_idx_XE = 2 * g_j + 1

                        if np.max(self.undo_vectorize(p, i_idx)) == i:
                            i_idx_XE = 2 * g_i
                        else:
                            i_idx_XE = 2 * g_i + 1

                        XEXE[i_idx_XE, j_idx_XE] = t
                        XEXE[j_idx_XE, i_idx_XE] = t
            return XEXE

        # Calculate the Gamma matrix in the Jacobian
        def calc_GammaMinus(gamma, active_dirs):
            """Calculate Gamma^minus (as a function of gamma vector, active directions)
            """
            to_diag = [[g] * (ug.size - 1) for (g, ug) in zip(gamma, active_dirs.values())]
            return block_diag(*[i for gp in to_diag for i in gp])

        def calc_GammaBar(gamma, active_dirs):
            """Calculate Gamma^minus (as a function of gamma vector, active directions)
            """
            to_diag = [[g] * (ug.size) for (g, ug) in zip(gamma, active_dirs.values())]
            return block_diag(*[i for gp in to_diag for i in gp])

        ## UNTESTED
        ## Calculate the Jacobian as a function of t, and location parameters i,j
        def Jacobian(t, i, j):
            ## Tasks:
            ## 1. Compute Q(t) by replacing x_i*x_j with t
            Q_ = Q(t, i, j, p=self.p, XE=self.XE)
            ## 2. Compute U using GL file lines 179
            U_ = self.U
            ## 3. Compute U_bar (V) using GL file lines 143-161
            V_ = self.V
            ## 4. Compute Lambda using GL file compute_Lg()
            L_ = self.L
            ## 5. Compute GammaBar using GL file calc_GammaBar()
            G_ = calc_GammaBar(self.observed_opt_state, self.active_dirs)

            J_ = np.block([(Q_ @ G_ + L_) @ V_, Q_ @ U_])
            return np.linalg.det(J_)

        print(Jacobian(10, 1, 2))

    @staticmethod
    def gaussian(X,
                 weights,
                 randomizer_scale,
                 sigma,
                 ridge_term=0,
                 quadratic=None,
                 perturb=None):
        ## weights : np.ndarray
        ##           Feature weights for L-1 penalty. If a float,
        ##           it is broadcast to all features.
        ##           Either 1) a positive float;
        ##           or     2) a symmetric pxp matrix with 0 on diagonals.

        ## randomizer_scale: positive float
        ##           Scale/standard deviation of the randomization term

        ## perturb : np.ndarray or None
        ##           Random perturbation subtracted as a linear trace
        ##           term in the objective function.
        ##           If not none, should be a pxp matrix with
        ##           zeros along its diagonal
        p = X.shape[1]
        n = X.shape[0]

        X_aug = paired_group_lasso.augment_X(X)
        Y_aug = paired_group_lasso.augment_Y(X)

        groups, groups_to_vars, weights = paired_group_lasso.create_groups(n, p, weights)

        loglike = rr.glm.gaussian(X_aug, Y_aug, coef=1. / sigma ** 2, quadratic=quadratic)
        randomizer = randomization.isotropic_gaussian((X_aug.shape[1],), randomizer_scale)

        return paired_group_lasso(loglike=loglike,
                                  n=n,
                                  p=p,
                                  groups=groups,
                                  weights=weights,
                                  randomizer=randomizer,
                                  randomizer_scale=randomizer_scale,
                                  ridge_term=ridge_term,
                                  perturb=perturb)
