import os, numpy as np, pandas, statsmodels.api as sm
import time

import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso

def estimate_sigma(X, y, nstep=20, tol=1.e-4):

    old_sigma = 1.
    for itercount in range(nstep):

        random_Z = np.zeros(p)
        sel = selection(X, y, random_Z, sigma=old_sigma)
        lam, epsilon, active, betaE, cube, initial_soln = sel
        print("active", active.sum())
        ols_fit = sm.OLS(y, X[:, active]).fit()
        new_sigma = np.linalg.norm(ols_fit.resid) / np.sqrt(n - active.sum() - 1)


        print("estimated sigma", new_sigma, old_sigma)
        if np.fabs(new_sigma - old_sigma) < tol :
            sigma = new_sigma
            break
        old_sigma = new_sigma

    return sigma


path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
n, p = X.shape
print("dims", n,p)
X -= X.mean(0)[None, :]
X /= (X.std(0)[None, :] * np.sqrt(n))

y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
y = y.reshape((y.shape[0],))

sigma = estimate_sigma(X, y, nstep=20, tol=1.e-5)
#sigma = 0.37858429815791306
y /= sigma
tau = 1.

np.random.seed(4)
#random_Z = np.random.normal(loc=0.0, scale= 1., size= p)
random_Z = np.zeros(p)
sel = selection(X, y, random_Z, randomization_scale=tau)

lam, epsilon, active, betaE, cube, initial_soln = sel

print("value of tuning parameter",lam)
print("nactive", active.sum())
active_set = [i for i in range(p) if active[i]]
print("active variables", active_set)
print("initial lasso", betaE)

noise_variance = 1.
print("noise variance", noise_variance)
nactive = betaE.shape[0]
active_sign = np.sign(betaE)
feasible_point = np.fabs(betaE)
lagrange = lam * np.ones(p)

generative_X = X[:, active]
prior_variance = 1000.
randomizer = randomization.isotropic_gaussian((p,), tau)

Q = np.linalg.inv(prior_variance* (generative_X.dot(generative_X.T)) + noise_variance* np.identity(n))
post_mean = prior_variance * ((generative_X.T.dot(Q)).dot(y))
post_var = prior_variance* np.identity(nactive) - ((prior_variance**2)*(generative_X.T.dot(Q).dot(generative_X)))
unadjusted_intervals = np.vstack([post_mean - 1.65*np.sqrt(post_var.diagonal()),post_mean + 1.65*np.sqrt(post_var.diagonal())])
print("unadjusted estimates", (sigma* unadjusted_intervals).T)

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
map = inf.map_solve(nstep = 100)[::-1]
print("selective map", sigma* map[1])

toc = time.time()
samples = inf.posterior_samples()
tic = time.time()
print('sampling time', tic - toc)


adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
sel_mean = np.mean(samples, axis=0)

print("active variables", active_set)
print("\n")
print("unadjusted estimates", sigma* post_mean, sigma* unadjusted_intervals)
print("\n")
print("adjusted_intervals", sigma* sel_mean, sigma* adjusted_intervals)

ad_length = sigma*(adjusted_intervals[1, :] - adjusted_intervals[0, :])
unad_length = sigma*(unadjusted_intervals[1, :] - unadjusted_intervals[0, :])

print("\n")
print("unadjusted and adjusted lengths", unad_length.sum()/nactive, ad_length.sum()/nactive)
