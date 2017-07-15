import os, numpy as np, pandas, statsmodels.api as sm
import time
import sys

import regreg.api as rr
#from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
#from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
#    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.ridge_target import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.estimator import M_estimator_exact

def selection(X, y, random_Z, randomization_scale=1, sigma=None, method="theoretical"):
    n, p = X.shape
    loss = rr.glm.gaussian(X,y)
    epsilon = 1. / np.sqrt(n)
    lam_frac = 1.
    if sigma is None:
        sigma = 1.
    if method == "theoretical":
        lam = 1. * sigma * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 10000)))).max(0))

    W = np.ones(p)*lam
    penalty = rr.group_lasso(np.arange(p), weights = dict(zip(np.arange(p), W)), lagrange=1.)

    # initial solution

    problem = rr.simple_problem(loss, penalty)
    random_term = rr.identity_quadratic(epsilon, 0, -randomization_scale * random_Z, 0)


    solve_args = {'tol': 1.e-10, 'min_its': 100, 'max_its': 500}
    initial_soln = problem.solve(random_term, **solve_args)
    active = (initial_soln != 0)
    if np.sum(active) == 0:
        return None
    initial_grad = loss.smooth_objective(initial_soln, mode='grad')
    betaE = initial_soln[active]
    subgradient = -(initial_grad+epsilon*initial_soln-randomization_scale*random_Z)
    cube = subgradient[~active]/lam
    return lam, epsilon, active, betaE, cube, initial_soln

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma,
                           true_mean):

    from selection.api import randomization

    n, p = X.shape

    random_Z = np.random.standard_normal(p)
    sel = selection(X, y, random_Z)
    lam, epsilon, active, betaE, cube, initial_soln = sel

    if sel is not None:

        lagrange = lam * np.ones(p)
        active_sign = np.sign(betaE)
        nactive = active.sum()
        print("number of selected variables by Lasso", nactive)
        active_set = np.asarray([i for i in range(p) if active[i]])
        print("active set", active_set)

        projection_active = X[:, active].dot(
            np.linalg.inv(X[:, active].T.dot(X[:, active]) + 0.*epsilon * np.identity(nactive)))
        true_val = projection_active.T.dot(true_mean)
        print("true value", true_val)

        feasible_point = np.fabs(betaE)

        noise_variance = sigma ** 2

        randomizer = randomization.isotropic_gaussian((p,), 1.)

        generative_X = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active]))).dot\
            (X[:, active].T.dot(X[:, active]) + 0.*epsilon * np.identity(nactive))
        prior_variance = 1000.

        M_1 = prior_variance * (X[:, active].dot(X[:, active].T)) + noise_variance * np.identity(n)
        M_2 = prior_variance * ((X[:, active].dot(X[:, active].T)).dot(projection_active))
        M_3 = prior_variance * (projection_active.T.dot(X[:, active].dot(X[:, active].T)).dot(projection_active))
        post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

        print("observed data", post_mean)

        post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

        unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                          post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

        print("unadjusted intervals", unadjusted_intervals)

        coverage_unad = np.zeros(nactive)
        unad_length = np.zeros(nactive)

        for l in range(nactive):
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[l] += 1
            unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]

        print("coverage unad", coverage_unad)

        bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

        naive_cov = coverage_unad.sum() / nactive

        unad_len = unad_length.sum()/nactive

        return np.vstack([naive_cov, unad_len , bayes_risk_unad])


class generate_data():

    def __init__(self, X, sigma=1., signals= "None", model = "Bayesian"):
         (self.sigma, self.signals) = (sigma, signals)

         self.n, self.p = X.shape
         X -= X.mean(0)[None, :]
         X /= (X.std(0)[None, :] * np.sqrt(n))

         self.X = X
         beta_true = np.zeros(self.p)

         #print("correlation of positions", (abs(np.corrcoef(X.T))[3283,]))

         if model is "Bayesian":
             s = 0
             u = np.random.uniform(0.,1.,self.p)
             for i in range(p):
                 if u[i]<= 0.95:
                     beta_true[i] = np.random.laplace(loc=0., scale= 0.1)
                 else:
                     beta_true[i] = np.random.laplace(loc=0., scale= 1.)
         else:
             if self.signals is None:
                 beta_true = np.zeros(self.p)
             else:
                 beta_true[self.signals] = 10.

         self.beta = beta_true

         print("true beta",self.beta[self.signals])

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X.dot(self.beta), Y, self.beta * self.sigma, self.sigma

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))


if __name__ == "__main__":

    path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

    X_unpruned = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
    n, p = X_unpruned.shape

    prototypes = np.loadtxt("/Users/snigdhapanigrahi/Results_bayesian/Egene_data/prototypes.txt", delimiter='\t')
    prototypes = prototypes.astype(int)
    prototypes = prototypes -1
    print("prototypes", prototypes.shape[0])

    X = X_unpruned[:, prototypes]
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))

    X_transposed = unique_rows(X.T)
    X = X_transposed.T

    n, p = X.shape
    print("dims", n, p)

    signals = np.loadtxt("/Users/snigdhapanigrahi/Results_bayesian/Egene_data/signal_3.txt", delimiter='\t')
    signals = signals.astype(int)

    print("shape of signals", signals.shape)

    niter = 50

    unad_cov = 0.
    unad_len = 0.
    unad_risk = 0.

    for i in range(niter):
        ### GENERATE Y BASED ON SEED
        np.random.seed(i)  # ensures different y

        #sample = generate_data(X_unpruned, sigma=1., signals=signals[i,:] - 1, model="Frequentist")
        sample = generate_data(X_unpruned, sigma=1., signals=None, model="Frequentist")

        true_mean, y, beta, sigma = sample.generate_response()

        ### RUN LASSO AND TEST
        lasso = randomized_lasso_trial(X,
                                       y,
                                       beta,
                                       sigma,
                                       true_mean)

        if lasso is not None:
            unad_cov += lasso[0, 0]
            unad_len += lasso[1, 0]
            unad_risk += lasso[2, 0]
            print("\n")
            print("iteration completed", i)
            print("\n")
            print("unadjusted coverage", unad_cov)
            print("unadjusted lengths", unad_len)
            print("unadjusted risks", unad_risk)


    niter = niter
    print("unadjusted coverage", unad_cov / niter)
    print("unadjusted lengths", unad_len / niter)
    print("unadjusted risks", unad_risk / niter)
