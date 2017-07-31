import os, numpy as np, pandas, statsmodels.api as sm
import time
import sys

import regreg.api as rr
from selection.algorithms.lasso import lasso
#from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
#from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
#    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.ridge_target import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.estimator import M_estimator_exact
from scipy.stats import norm as ndist
from scipy.optimize import bisect

def restricted_gaussian(Z, interval=[-5.,5.]):
    L_restrict, U_restrict = interval
    Z_restrict = max(min(Z, U_restrict), L_restrict)
    return ((ndist.cdf(Z_restrict) - ndist.cdf(L_restrict)) /
            (ndist.cdf(U_restrict) - ndist.cdf(L_restrict)))

def pivot(L_constraint, Z, U_constraint, S, truth=0):
    F = restricted_gaussian
    if F((U_constraint - truth) / S) != F((L_constraint -  truth) / S):
        v = ((F((Z-truth)/S) - F((L_constraint-truth)/S)) /
             (F((U_constraint-truth)/S) - F((L_constraint-truth)/S)))
    elif F((U_constraint - truth) / S) < 0.1:
        v = 1
    else:
        v = 0
    return v

def equal_tailed_interval(L_constraint, Z, U_constraint, S, alpha=0.05):

    lb = Z - 5 * S
    ub = Z + 5 * S

    def F(param):
        return pivot(L_constraint, Z, U_constraint, S, truth=param)

    FL = lambda x: (F(x) - (1 - 0.5 * alpha))
    FU = lambda x: (F(x) - 0.5* alpha)
    L_conf = bisect(FL, lb, ub)
    U_conf = bisect(FU, lb, ub)
    return np.array([L_conf, U_conf])

def lasso_Gaussian(X, y, lam, true_mean):

    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()


    active = soln != 0
    print("Lasso estimator", soln[active])
    nactive = active.sum()
    print("nactive", nactive)

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))
    true_val = projection_active.T.dot(true_mean)

    active_set = np.nonzero(active)[0]
    print("active set", active_set)

    active_signs = np.sign(soln[active])
    C = L.constraints
    sel_intervals = np.zeros((nactive, 2))

    coverage_ad = np.zeros(nactive)
    ad_length = np.zeros(nactive)

    if C is not None:
        one_step = L.onestep_estimator
        print("one step", one_step)
        for i in range(one_step.shape[0]):
            eta = np.zeros_like(one_step)
            eta[i] = active_signs[i]
            alpha = 0.1

            if C.linear_part.shape[0] > 0:  # there were some constraints
                L, Z, U, S = C.bounds(eta, one_step)
                _pval = pivot(L, Z, U, S)
                # two-sided
                _pval = 2 * min(_pval, 1 - _pval)

                L, Z, U, S = C.bounds(eta, one_step)
                _interval = equal_tailed_interval(L, Z, U, S, alpha=alpha)
                _interval = sorted([_interval[0] * active_signs[i],
                                    _interval[1] * active_signs[i]])

            else:
                obs = (eta * one_step).sum()
                sd = np.sqrt((eta * C.covariance.dot(eta)))
                Z = obs / sd
                _pval = 2 * (ndist.sf(min(np.fabs(Z))) - ndist.sf(5)) / (ndist.cdf(5) - ndist.cdf(-5))

                _interval = (obs - ndist.ppf(1 - alpha / 2) * sd,
                             obs + ndist.ppf(1 - alpha / 2) * sd)

            sel_intervals[i, :] = _interval

            if (sel_intervals[i, 0] <= true_val[i]) and (true_val[i] <= sel_intervals[i, 1]):
                coverage_ad[i] += 1

            ad_length[i] = sel_intervals[i, 1] - sel_intervals[i, 0]

        sel_cov = coverage_ad.sum() / nactive
        ad_len = ad_length.sum() / nactive
        #ad_risk = np.power(soln[active] - true_val, 2.).sum() / nactive
        ad_risk = np.power(one_step - true_val, 2.).sum() / nactive

        return sel_cov, ad_len, ad_risk

    else:

        return 0.,0.,0.



class generate_data():

    def __init__(self, X, sigma=1., signals= "None", model = "Bayesian"):
         (self.sigma, self.signals) = (sigma, signals)

         self.n, self.p = X.shape
         X -= X.mean(0)[None, :]
         X /= (X.std(0)[None, :] * np.sqrt(n))

         self.X = X
         beta_true = np.zeros(self.p)
         print("here")
         #print("correlation of positions", (abs(np.corrcoef(X.T))[3283,]))

         if model is "Bayesian":
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
                 print("here")
                 beta_true[self.signals] = 2.5

         self.beta = beta_true

         print("true beta", self.beta[self.signals])

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
    prototypes = prototypes.astype(int)-1
    print("prototypes", prototypes.shape[0])

    signals = np.loadtxt("/Users/snigdhapanigrahi/Results_bayesian/Egene_data/signal_3.txt", delimiter='\t')
    signals = signals.astype(int)
    print("signals", signals)

    X = X_unpruned[:, prototypes]
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))

    X_transposed = unique_rows(X.T)
    X = X_transposed.T

    n, p = X.shape
    print("dims", n, p)

    niter = 50

    ad_cov = 0.
    unad_cov = 0.
    ad_len = 0.
    unad_len = 0.
    ad_risk = 0.
    unad_risk = 0.
    lam_frac = 0.75
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

    for i in range(niter):

        np.random.seed(i+1)  # ensures different y
        sample = generate_data(X_unpruned, sigma=1., signals= signals[i,:]-1, model="Frequentist")

        true_mean, y, beta, sigma = sample.generate_response()

        ### RUN LASSO

        lasso_results = lasso_Gaussian(X,
                                       y,
                                       lam,
                                       true_mean)

        ad_cov += lasso_results[0]
        ad_len += lasso_results[1]
        ad_risk += lasso_results[2]
        print("\n")
        print("iteration completed", i)
        print("\n")
        print("adjusted coverage", ad_cov)
        print("adjusted lengths", ad_len)
        print("adjusted risks", ad_risk)

    print("adjusted coverage", ad_cov / niter)
    print("adjusted lengths", ad_len / niter)
    print("adjusted risks", ad_risk / niter)




