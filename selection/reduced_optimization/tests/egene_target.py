import os, numpy as np, pandas, statsmodels.api as sm
import time
import sys

import regreg.api as rr
from selection.bayesian.initial_soln import selection
from selection.randomized.api import randomization
#from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
#    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.lasso_reduced import nonnegative_softmax_scaled, neg_log_cube_probability, selection_probability_lasso, \
    sel_prob_gradient_map_lasso, selective_inf_lasso
from selection.reduced_optimization.estimator import M_estimator_exact

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

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
X -= X.mean(0)[None, :]
X /= (X.std(0)[None, :] * np.sqrt(n))

X_transposed = unique_rows(X.T)
X = X_transposed.T
n, p = X.shape
print("dims", n,p)

y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
y = y.reshape((y.shape[0],))

sigma = estimate_sigma(X, y, nstep=20, tol=1.e-5)
y /= sigma

lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * 1.
loss = rr.glm.gaussian(X, y)

epsilon = 1. / np.sqrt(n)

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p),
                         weights=dict(zip(np.arange(p), W)), lagrange=1.)

np.random.seed(4)
randomization = randomization.isotropic_gaussian((p,), scale=1.)

M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer='gaussian')

M_est.solve_approx()
active = M_est._overall
print("active sum", active[1964])
active_set = np.asarray([i for i in range(p) if active[i]])
nactive = np.sum(active)
betaE = M_est.initial_soln[M_est._overall]

noise_variance = 1.
active_sign = np.sign(betaE)
feasible_point = np.fabs(betaE)
lagrange = lam * np.ones(p)

sys.stderr.write("number of active selected by lasso" + str(nactive) + "\n")
sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
active_1 = M_est._overall.copy()
active_1[1964] = False
active_1[1458] = False
active_1[3714] = False
active_1[3728] = False

corr_X = np.corrcoef(X[:,active_1].T)
indices_corr = np.transpose((np.abs(corr_X)>0.7).nonzero())
print("Correlation structure", indices_corr)

generative_X = X[:, active_1]
prior_variance = 100000000.
nactive_1 = nactive -4

projection_active = X[:, active_1].dot(np.linalg.inv(X[:, active_1].T.dot(X[:, active_1])))
M_1 = prior_variance * (X[:,active].dot(X[:,active].T)) + noise_variance * np.identity(n)
M_2 = prior_variance * ((X[:,active].dot(X[:,active].T)).dot(projection_active))
M_3 = prior_variance * (projection_active.T.dot(X[:,active].dot(X[:,active].T)).dot(projection_active))
post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y)

post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                  post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])
print("unadjusted estimates", sigma*post_mean, sigma*(unadjusted_intervals).T)

print("dims", X.shape, feasible_point.shape, active.sum(), generative_X.shape)

grad_map = sel_prob_gradient_map_lasso(X,
                                       feasible_point,
                                       active,
                                       active_sign,
                                       lagrange,
                                       generative_X,
                                       noise_variance,
                                       randomization,
                                       epsilon)

initial_soln = M_est.initial_soln[active_1]
inf = selective_inf_lasso(y, grad_map, prior_variance, initial_soln)
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



adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])
