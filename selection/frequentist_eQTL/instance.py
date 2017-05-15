import numpy as np

class instance(object):

    def __init__(self, n, p, s, snr=5., sigma=1., rho=0, random_signs=False, scale =True, center=True):
         (self.n, self.p, self.s,
         self.sigma,
         self.rho) = (n, p, s,
                     sigma,
                     rho)

         self.X = (np.sqrt(1 - self.rho) * np.random.standard_normal((self.n, self.p)) +
              np.sqrt(self.rho) * np.random.standard_normal(self.n)[:, None])
         if center:
             self.X -= self.X.mean(0)[None, :]
         if scale:
             self.X /= (self.X.std(0)[None, :] * np.sqrt(self.n))

         self.beta = np.zeros(p)

         if snr == 5.:
             if s==5:
                 self.beta[:self.s] = np.linspace(0.5, 5., num=s)
             elif s==4:
                 self.beta[:self.s] = np.linspace(1.0625, 4.4375, num=s)
             elif s==3:
                 self.beta[:self.s] = np.linspace(1.625, 3.875, num=s)
             elif s==2:
                 self.beta[:self.s] = np.linspace(2.1875, 3.3125, num=s)
             elif s==1:
                 self.beta[:self.s] = 2.75
         elif snr == 3.5:
             if s==5:
                 self.beta[:self.s] = np.linspace(0.5, 3.5, num=s)
             elif s==4:
                 self.beta[:self.s] = np.linspace(0.875, 3.125, num=s)
             elif s==3:
                 self.beta[:self.s] = np.linspace(1.25, 2.75, num=s)
             elif s==2:
                 self.beta[:self.s] = np.linspace(1.625, 2.375, num=s)
             elif s==1:
                 self.beta[:self.s] = 2.

         if random_signs:
             self.beta[:self.s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
         self.active = np.zeros(p, np.bool)
         self.active[:self.s] = True

    def _noise(self):
        return np.random.standard_normal(self.n)

    def generate_response(self):

        Y = (self.X.dot(self.beta) + self._noise()) * self.sigma
        return self.X, Y, self.beta * self.sigma, np.nonzero(self.active)[0], self.sigma