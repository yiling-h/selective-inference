import numpy as np

from .MLE import selective_MLE
from ..constraints.affine import constraints

class gaussian_query():
    useC = True

    """
    A class with Gaussian perturbation to the objective.
    """

    def fit(self, perturb=None):

        p = self.nfeature

        # take a new perturbation if supplied
        if perturb is not None:
            self._initial_omega = perturb
        if self._initial_omega is None:
            self._initial_omega = self.randomizer.sample()

    # Private methods

    def _setup_query(self,
                     linear_part,
                     offset,
                     opt_linear,
                     opt_offset,
                     dispersion=1):

        A, b = linear_part, offset
        if not np.all(A.dot(self.observed_opt_state) - b <= 0):
            raise ValueError('constraints not satisfied')

        _, randomizer_prec = self.randomizer.cov_prec
        randomizer_prec = randomizer_prec / dispersion

        self.randomizer_prec = randomizer_prec

        (cond_mean,
         cond_cov,
         cond_precision,
         logdens_linear) = self._setup_implied_gaussian(opt_linear,
                                                        opt_offset,
                                                        dispersion)

        self.cond_mean, self.cond_cov, self.logdens_linear = cond_mean, cond_cov, logdens_linear

        self.affine_con = constraints(A,
                                      b,
                                      mean=cond_mean,
                                      covariance=cond_cov)

        self.score_offset = self.observed_score_state + opt_offset

    def _setup_implied_gaussian(self,
                                opt_linear,
                                opt_offset,
                                # optional dispersion parameter
                                # for covariance of randomization
                                dispersion=1):

        if np.asarray(self.randomizer_prec).shape in [(), (0,)]:
            cond_precision = opt_linear.T.dot(opt_linear) * self.randomizer_prec
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T) * self.randomizer_prec
        else:
            cond_precision = opt_linear.T.dot(self.randomizer_prec.dot(opt_linear))
            cond_cov = np.linalg.inv(cond_precision)
            logdens_linear = cond_cov.dot(opt_linear.T).dot(self.randomizer_prec)

        cond_mean = -logdens_linear.dot(self.observed_score_state + opt_offset)

        return cond_mean, cond_cov, cond_precision, logdens_linear

    def selective_MLE_inference(self,
                                observed_target,
                                target_cov,
                                target_score_cov,
                                level=0.9,
                                solve_args={'tol': 1.e-12}):
        """

        Parameters
        ----------

        observed_target : ndarray
            Observed estimate of target.

        target_cov : ndarray
            Estimated covaraince of target.

        target_score_cov : ndarray
            Estimated covariance of target and score of randomized query.

        level : float, optional
            Confidence level.

        solve_args : dict, optional
            Arguments passed to solver.

        """

        return selective_MLE(observed_target,
                             target_cov,
                             target_score_cov,
                             self.observed_opt_state,
                             self.cond_mean,
                             self.cond_cov,
                             self.logdens_linear,
                             self.affine_con.linear_part,
                             self.affine_con.offset,
                             self.randomizer_prec,
                             self.score_offset,
                             solve_args=solve_args,
                             level=level)


