import os, numpy as np, pandas, statsmodels.api as sm
import time
import sys

import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
#from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
#    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.ridge_target import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.estimator import M_estimator_exact

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma):

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
            np.linalg.inv(X[:, active].T.dot(X[:, active]) + 0.01*epsilon * np.identity(nactive)))
        true_val = projection_active.T.dot(X.dot(beta))
        print("true value", true_val)

        feasible_point = np.fabs(betaE)

        noise_variance = sigma ** 2

        randomizer = randomization.isotropic_gaussian((p,), 1.)

        generative_X = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active]))).dot\
            (X[:, active].T.dot(X[:, active]) + 0.01*epsilon * np.identity(nactive))
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

        coverage_ad = np.zeros(nactive)
        coverage_unad = np.zeros(nactive)
        ad_length = np.zeros(nactive)
        unad_length = np.zeros(nactive)

        for l in range(nactive):
            if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
                coverage_unad[l] += 1
            unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]

        print("coverage unad", coverage_unad)

        grad_map = sel_prob_gradient_map_lasso(X,
                                               feasible_point,
                                               active,
                                               active_sign,
                                               lagrange,
                                               generative_X,
                                               noise_variance,
                                               randomizer,
                                               epsilon)

        inf = selective_inf_lasso(y, grad_map, prior_variance)

        samples = inf.posterior_samples()

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        selective_mean = np.mean(samples, axis=0)

        print("adjusted intervals", adjusted_intervals)

        for l in range(nactive):
            if (adjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= adjusted_intervals[1, l]):
                coverage_ad[l] += 1
            ad_length[l] = adjusted_intervals[1, l] - adjusted_intervals[0, l]

        sel_cov = coverage_ad.sum() / nactive
        naive_cov = coverage_unad.sum() / nactive
        ad_len = ad_length.sum() / nactive
        unad_len = unad_length.sum() / nactive
        bayes_risk_ad = np.power(selective_mean - true_val, 2.).sum() / nactive
        bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

        print("results", sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad)
        return np.vstack([sel_cov, naive_cov, ad_len, unad_len, bayes_risk_ad, bayes_risk_unad])

    else:
        return None

class generate_data():

    def __init__(self, X, sigma=1., s=5, model = "Bayesian"):
         (self.X, self.sigma) = (X, sigma)

         self.n, self.p = self.X.shape
         beta_true = np.zeros(p)

         #print("correlation of positions", (abs(np.corrcoef(X.T))[3283,]))

         if model is "Bayesian":
             s = 0
             u = np.random.uniform(0.,1.,p)
             for i in range(p):
                 if u[i]<= 0.95:
                     beta_true[i] = np.random.laplace(loc=0., scale= 0.1)
                 else:
                     beta_true[i] = np.random.laplace(loc=0., scale= 1.)
         else:
             #u = np.random.choice(p, s)
             #u = np.linspace(0,p,5,endpoint=False).astype(int)
             u = np.array([0,1153])
             print("True positions of signals", u)
             beta_true[0] = 10.
             beta_true[1153] = -10.
             #beta_true[3283] = 10.

         self.beta = beta_true
         print("correlation of positions", np.corrcoef(X[:, u].T))

    def generate_response(self):

        Y = (self.X.dot(self.beta) + np.random.standard_normal(self.n)) * self.sigma

        return self.X, Y, self.beta * self.sigma, self.sigma

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

if __name__ == "__main__":

    #np.random.seed(2)
    path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

    X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))

    X_transposed = unique_rows(X.T)
    X = X_transposed.T

    n, p = X.shape
    print("dims", n, p)

    sample = generate_data(X, sigma=1., s=5, model = "Frequentist")

    niter = 1

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.

    for i in range(niter):

         ### GENERATE Y BASED ON SEED
         np.random.seed(i+7)  # ensures different y
         X, y, beta, sigma = sample.generate_response()

         print("true mean", X.dot(beta))
         ### RUN LASSO AND TEST
         lasso = randomized_lasso_trial(X,
                                        y,
                                        beta,
                                        sigma)

         if lasso is not None:
             ad_cov += lasso[0,0]
             unad_cov += lasso[1,0]
             ad_len += lasso[2, 0]
             unad_len += lasso[3, 0]
             ad_risk += lasso[4, 0]
             unad_risk += lasso[5, 0]
             print("\n")
             print("iteration completed", i)
             print("\n")
             print("adjusted and unadjusted coverage", ad_cov, unad_cov)
             print("adjusted and unadjusted lengths", ad_len, unad_len)
             print("adjusted and unadjusted risks", ad_risk, unad_risk)
