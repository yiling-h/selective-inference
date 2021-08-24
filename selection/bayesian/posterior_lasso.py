import numpy as np, sys
from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from scipy.stats import norm as ndist
from scipy.linalg import fractional_matrix_power

class langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 stepsize,
                 max_jump = 5.e+1):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)

        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)
        self.sample = np.copy(initial_condition)
        self.max_jump = max_jump


    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):

        while True:
            grad_posterior = self.gradient_map(self.state)
            candidate = (self.state + self.stepsize * grad_posterior[0]
                         + np.sqrt(2.) * self._noise.rvs(self._shape) * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate)[0])):
                print(candidate, self._sqrt_step, grad_posterior[0])
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                self.sample[:] = grad_posterior[1]
                print(" next sample ", self.state[:], self.sample[:])
                break

class MA_langevin(object):

    def __init__(self,
                 initial_condition,
                 gradient_map,
                 stepsize,
                 max_jump = 5.e+1):

        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                                 gradient_map,
                                 stepsize)

        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0,scale=1)
        self.sample = np.copy(initial_condition)
        self.max_jump = max_jump

        posterior_old = self.gradient_map(self.state)
        self.gradient_old = posterior_old[0]
        self.postvalue_old = posterior_old[2]
        self.reparam_old = posterior_old[1]

    def __iter__(self):
        return self

    def next(self):
        while True:
            candidate = (self.state + self.stepsize * self.gradient_old
                        + np.sqrt(2.)* self._noise.rvs(self._shape) * self._sqrt_step)

            posterior_current = self.gradient_map(candidate)
            diff = np.linalg.norm(candidate - self.state - self.stepsize * self.gradient_old)**2. -\
                   np.linalg.norm(self.state - candidate - self.stepsize * posterior_current[0])**2.
            accept_reject = min(1., np.exp(((1./(4. * self.stepsize))* diff) + (posterior_current[2]-self.postvalue_old)))
            print("check", accept_reject, ((1./(4. * self.stepsize))* diff), (posterior_current[2]-self.postvalue_old))

            if np.random.uniform(0., 1.)> accept_reject:
                continue
            else:
                self.state[:] = candidate
                self.sample[:] = self.reparam_old
                print(" next sample ", self.state[:], self.sample[:])
                self.gradient_old[:] = posterior_current[0]
                self.postvalue_old = posterior_current[2]
                self.reparam_old[:] = posterior_current[1]
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

    def prior(self, target_parameter, prior_var=100.):
        grad_prior = -target_parameter/prior_var
        log_prior = -np.linalg.norm(target_parameter)/(2.*prior_var)
        return grad_prior, log_prior

    def det_initial_point(self, initial_soln, solve_args={'tol':1.e-12}):

        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(initial_soln) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_C

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        initial_point = initial_soln + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
        return initial_point

    def gradient_log_likelihood(self, target_parameter, solve_args={'tol':1.e-15}):

        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(target_parameter) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        reparam = target_parameter + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
        neg_normalizer = (target_parameter - reparam).T.dot(prec_target).dot(target_parameter - reparam)/2. + val + mean_opt.T.dot(prec_opt).dot(mean_opt) / 2.

        grad_barrier = np.diag(2. / ((1. + soln) ** 3.) - 2. / (soln ** 3.))

        L = target_lin.T.dot(prec_opt)
        N = L.dot(hess)
        jacobian = (np.identity(observed_target.shape[0]) + self.cov_target.dot(L).dot(target_lin)) - \
                   self.cov_target.dot(N).dot(L.T)

        log_lik = -((observed_target - reparam).T.dot(prec_target).dot(observed_target - reparam)) / 2. + neg_normalizer \
                  + np.log(np.linalg.det(jacobian))

        grad_lik = jacobian.T.dot(prec_target).dot(observed_target)
        grad_neg_normalizer = -jacobian.T.dot(prec_target).dot(target_parameter)

        opt_num = self.cond_cov.shape[0]
        grad_jacobian = np.zeros(opt_num)
        A = np.linalg.inv(jacobian).dot(self.cov_target).dot(N)
        for j in range(opt_num):
            M = grad_barrier.dot(np.diag(N.T[:, j]))
            grad_jacobian[j] = np.trace(A.dot(M).dot(N.T))

        prior_info = self.prior(reparam)
        return grad_lik + grad_neg_normalizer + grad_jacobian + jacobian.T.dot(prior_info[0]), reparam, log_lik + prior_info[1]

    def posterior_sampler(self, nsample= 2000, nburnin=100, step=1., start=None, Metropolis=False):
        if start is None:
            start = self.det_initial_point(self.ini_estimate)
        state = start

        restart = True
        count = 0
        while restart == True:
            stepsize = 1. / (step * self.target_size)
            if Metropolis is False:
                sampler = langevin(state, self.gradient_log_likelihood, stepsize)

            else:
                sampler = MA_langevin(state, self.gradient_log_likelihood, stepsize)

            samples = np.zeros((nsample, self.target_size))
            if count == 2:
                raise ValueError('sampler escaping')
            for i in range(nsample):
                sampler.next()
                next_sample = sampler.sample.copy()
                if max(np.fabs(next_sample)) > 500. and i > nburnin:
                    step /= 0.50
                    restart = True
                    count += 1
                    break
                else:
                    restart = False
                    samples[i, :] = sampler.sample.copy()
                    sys.stderr.write("sample number: " + str(i) + "\n")

        return samples[nburnin:, :], count


class inference_lasso_hierarchical():

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

    def hierarchical_prior(self, target_parameter, var_parameter, lam):
        std_parameter = np.sqrt(var_parameter)
        grad_prior_par = -np.true_divide(target_parameter,  var_parameter)
        grad_prior_std = np.true_divide(target_parameter**2. , 2.*(var_parameter**2))- (lam/2.)-1./(2.*var_parameter)
        log_prior = -(np.linalg.norm(target_parameter)**2.) / (2.*var_parameter) - (lam * (np.linalg.norm(std_parameter)**2)/2.)-np.log(std_parameter)
        return grad_prior_par, grad_prior_std, log_prior

    def det_initial_point(self, initial_soln, solve_args={'tol':1.e-12}):

        if np.asarray(self.observed_target).shape in [(), (0,)]:
            raise ValueError('no target specified')

        observed_target = np.atleast_1d(self.observed_target)
        prec_target = np.linalg.inv(self.cov_target)

        target_lin = - self.logdens_linear.dot(self.cov_target_score.T.dot(prec_target))
        target_offset = self.cond_mean - target_lin.dot(observed_target)

        prec_opt = np.linalg.inv(self.cond_cov)
        mean_opt = target_lin.dot(initial_soln) + target_offset
        conjugate_arg = prec_opt.dot(mean_opt)

        solver = solve_barrier_affine_py

        val, soln, hess = solver(conjugate_arg,
                                 prec_opt,
                                 self.feasible_point,
                                 self.linear_part,
                                 self.offset,
                                 **solve_args)

        initial_point = initial_soln + self.cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
        return initial_point

    def gradient_log_likelihood(self, parameters, solve_args={'tol':1.e-15}):

        npar = self.target_size
        target_parameter = parameters[:npar]
        var_parameter = parameters[npar:]
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
        neg_normalizer = (target_parameter - reparam).T.dot(prec_target).dot(target_parameter - reparam)/2. \
                         + val + mean_opt.T.dot(prec_opt).dot(mean_opt) / 2.

        grad_barrier = np.diag(2. / ((1. + soln) ** 3.) - 2. / (soln ** 3.))

        L = target_lin.T.dot(prec_opt)
        N = L.dot(hess)
        jacobian = (np.identity(observed_target.shape[0]) + self.cov_target.dot(L).dot(target_lin)) - \
                   self.cov_target.dot(N).dot(L.T)

        log_lik = -((observed_target - reparam).T.dot(prec_target).dot(observed_target - reparam)) / 2. + neg_normalizer \
                  + np.log(np.linalg.det(jacobian))

        grad_lik = jacobian.T.dot(prec_target).dot(observed_target)
        grad_neg_normalizer = -jacobian.T.dot(prec_target).dot(target_parameter)

        opt_num = self.cond_cov.shape[0]
        grad_jacobian = np.zeros(opt_num)
        A = np.linalg.inv(jacobian).dot(self.cov_target).dot(N)
        for j in range(opt_num):
            M = grad_barrier.dot(np.diag(N.T[:, j]))
            grad_jacobian[j] = np.trace(A.dot(M).dot(N.T))

        prior_info = self.hierarchical_prior(reparam, var_parameter, lam=0.01)
        return np.append(grad_lik + grad_neg_normalizer + grad_jacobian + jacobian.T.dot(prior_info[0]), prior_info[1]),\
               np.append(reparam, var_parameter), log_lik + prior_info[2]

    def posterior_sampler(self, nsample= 2000, nburnin=100, step=1., start=None, Metropolis=False):
        if start is None:
            start = self.det_initial_point(self.ini_estimate)
        state = np.append(start, np.ones(self.target_size))

        restart = True
        count = 0
        while restart == True:
            stepsize = 1. / (step * (2*self.target_size))
            if Metropolis is False:
                sampler = langevin(state, self.gradient_log_likelihood, stepsize)

            else:
                sampler = MA_langevin(state, self.gradient_log_likelihood, stepsize)

            samples = np.zeros((nsample, 2*self.target_size))
            if count == 2:
                raise ValueError('sampler escaping')
                break
            for i in range(nsample):
                sampler.next()
                next_sample = sampler.sample.copy()
                if max(np.fabs(next_sample)[:self.target_size]) > 500. and i > nburnin:
                    step /= 0.50
                    restart = True
                    count += 1
                    break
                else:
                    restart = False
                    samples[i, :] = sampler.sample.copy()
                    sys.stderr.write("sample number: " + str(i) + "\n")

        return samples[nburnin:, :], count



def solve_barrier_affine_py(conjugate_arg,
                            precision,
                            feasible_point,
                            con_linear,
                            con_offset,
                            step=1,
                            nstep=2000,
                            min_its=500,
                            tol=1.e-12):

    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. \
                          + np.log(1.+ 1./((con_offset - con_linear.dot(u))/ scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) - con_linear.T.dot(1./(scaling + con_offset - con_linear.dot(u)) -
                                                                       1./(con_offset - con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(u))**2.)
                                                 + 1./((con_offset-con_linear.dot(u))**2.))).dot(con_linear)

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset-con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')

        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + barrier_hessian(current))
    return current_value, current, hess







