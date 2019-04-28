import numpy as np, sys

from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from scipy.stats import norm as ndist

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.vector(lambda)
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lambda, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)

    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    return estimate

def log_likelihood(target_parameter,
                   observed_target,
                   cov_target,
                   cov_target_score,
                   feasible_point,
                   cond_mean,
                   cond_cov,
                   logdens_linear,
                   linear_part,
                   offset,
                   solve_args={'tol': 1.e-12}):

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)

    target_lin = - logdens_linear.dot(cov_target_score.T.dot(prec_target))
    target_offset = cond_mean - target_lin.dot(observed_target)

    prec_opt = np.linalg.inv(cond_cov)
    mean_opt = target_lin.dot(target_parameter)+target_offset
    conjugate_arg = prec_opt.dot(mean_opt)

    solver = solve_barrier_affine_C

    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             feasible_point,
                             linear_part,
                             offset,
                             **solve_args)

    reparam = target_parameter + cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
    neg_normalizer = (target_parameter-reparam).T.dot(prec_target).dot(target_parameter-reparam)+ val + mean_opt.T.dot(prec_opt).dot(mean_opt)/2.

    L = target_lin.T.dot(prec_opt)
    jacobian = (np.identity(observed_target.shape[0])+ cov_target.dot(L).dot(target_lin)) - \
               cov_target.dot(L).dot(hess).dot(L.T)

    log_lik = -(observed_target-target_parameter).T.dot(prec_target).dot(observed_target-target_parameter)/2. + neg_normalizer \
              + np.log(np.linalg.det(jacobian))

    return log_lik

class projected_langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 stepsize):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)
        self.sample = np.copy(initial_condition)

    def __iter__(self):
        return self

    def next(self):
        while True:
            grad_posterior = self.gradient_map(self.state)
            candidate = (self.state + self.stepsize * grad_posterior[0]
                        + np.sqrt(2.)* self._noise.rvs(self._shape) * self._sqrt_step)
            if not np.all(np.isfinite(self.gradient_map(candidate)[0])):
                print(candidate, self._sqrt_step)
                self._sqrt_step *= 0.8
            else:
                self.state[:] = candidate
                self.sample[:] = grad_posterior[1]
                print(" next sample ", self.state[:], self.sample[:])
                break

class inference_lasso():

    def __init__(self,
                 observed_target,
                 cov_target,
                 cov_target_score,
                 feasible_point,
                 cond_mean,
                 cond_cov,
                 logdens_linear,
                 linear_part,
                 offset,
                 ini_estimate):

        self.observed_target = observed_target
        self.cov_target = cov_target
        self.cov_target_score = cov_target_score
        self.feasible_point = feasible_point
        self.cond_mean = cond_mean
        self.cond_cov = cond_cov
        self.target_size = cond_cov.shape[0]
        self.logdens_linear = logdens_linear
        self.linear_part = linear_part
        self.offset = offset
        self.ini_estimate = ini_estimate

    def det_initial_point(self, solve_args={'tol':1.e-12}):

        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(self.ini_estimate) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_C

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        initial_point = self.ini_estimate + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
        return initial_point

    def gradient_log_likelihood(self, target_parameter, solve_args={'tol':1.e-12}):

        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(target_parameter) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_C

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        reparam = target_parameter + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))

        grad_barrier = np.diag(2. / ((1. + soln) ** 3.) - 2. / (soln ** 3.))

        L = target_lin.T.dot(prec_opt)
        N = L.dot(hess)
        jacobian = (np.identity(observed_target.shape[0]) + self.cov_target.dot(L).dot(target_lin)) - \
                   self.cov_target.dot(N).dot(L.T)

        grad_lik = jacobian.T.dot(prec_target).dot(observed_target)
        grad_neg_normalizer = -jacobian.T.dot(prec_target).dot(target_parameter)

        opt_num = self.cond_cov.shape[0]
        grad_jacobian = np.zeros(opt_num)
        A = np.linalg.inv(jacobian).dot(self.cov_target).dot(N)
        for j in range(opt_num):
            M = grad_barrier.dot(np.diag(N.T[:, j]))
            grad_jacobian[j] = np.trace(A.dot(M).dot(N.T))

        return grad_lik + grad_neg_normalizer + grad_jacobian -target_parameter/100., reparam

    def posterior_sampler(self, nsample= 2000, nburnin=100):

        initial_state = self.det_initial_point()
        state = initial_state

        stepsize = 1. / (1 * self.target_size)
        sampler = projected_langevin(state, self.gradient_log_likelihood, stepsize)

        samples = []

        for i in range(nsample):
            sampler.next()
            samples.append(sampler.sample.copy())
            sys.stderr.write("sample number: " + str(i) + "\n")

        samples = np.array(samples)
        return samples[nburnin:, :]







