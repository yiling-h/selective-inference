import numpy as np
from scipy.stats import norm as ndist
from scipy.linalg import fractional_matrix_power

class langevin(object):


    def __init__(self,
                 initial_condition,
                 gradient_map,
                 proposal_scale,
                 stepsize,
                 scaling):

        """
               A prototype gradient-based sampler
               Parameters
               ----------
               initial_condition: initial sample
               gradient_map: gradient of log-posterior, value of log-posterior
               proposal_scale: covariance of a Gaussian proposal
               stepsize: stepsize of sampler
           """


        (self.state,
         self.gradient_map,
         self.stepsize) = (np.copy(initial_condition),
                           gradient_map,
                           stepsize)
        self.proposal_scale = proposal_scale
        self._shape = self.state.shape[0]
        self._sqrt_step = np.sqrt(self.stepsize)
        self._noise = ndist(loc=0, scale=1)
        self.sample = np.copy(initial_condition)
        self.scaling = scaling

        self.proposal_sqrt = fractional_matrix_power(self.proposal_scale, 0.5)

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):

        while True:

            self.grad_posterior = self.gradient_map(self.state, self.scaling)
            candidate = (self.state + self.stepsize * self.proposal_scale.dot(self.grad_posterior[0])
                         + np.sqrt(2.) * (self.proposal_sqrt.dot(self._noise.rvs(self._shape))) * self._sqrt_step)

            if not np.all(np.isfinite(self.gradient_map(candidate)[0])):
                self.stepsize *= 0.5
                self._sqrt_step = np.sqrt(self.stepsize)
            else:
                self.state[:] = candidate
                break

        return self.state
