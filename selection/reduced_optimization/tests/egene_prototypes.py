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
            np.linalg.inv(X[:, active].T.dot(X[:, active]) + 0.*epsilon * np.identity(nactive)))

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

        unad_length = ((unadjusted_intervals[1, :] - unadjusted_intervals[0, :]).sum()) / nactive
        #print("unadjusted intervals", unadjusted_intervals)

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

        #map = inf.map_solve(nstep=200)[::-1]

        #print("sel map", map)

        samples = inf.posterior_samples()

        adjusted_intervals = np.vstack([np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)])

        selective_mean = np.mean(samples, axis=0)

        ad_length = ((adjusted_intervals[1,:]- adjusted_intervals[0,:]).sum())/nactive
        #print("adjusted intervals", adjusted_intervals)

        return adjusted_intervals, unadjusted_intervals, selective_mean, post_mean, ad_length, unad_length

    else:

        return None

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

if __name__ == "__main__":

    #np.random.seed(2)
    path = '/Users/snigdhapanigrahi/Results_bayesian/Egene_data/'

    X = np.load(os.path.join(path + "X_" + "ENSG00000131697.13") + ".npy")
    n, p = X.shape

    prototypes = np.loadtxt("/Users/snigdhapanigrahi/Results_bayesian/Egene_data/prototypes.txt", delimiter='\t')
    prototypes = prototypes.astype(int)-1
    print("prototypes", prototypes.shape[0])

    X = X[:, prototypes]
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))

    X_transposed = unique_rows(X.T)
    X = X_transposed.T

    n, p = X.shape
    print("dims", n, p)

    y = np.load(os.path.join(path + "y_" + "ENSG00000131697.13") + ".npy")
    y = y.reshape((y.shape[0],))

    sigma = estimate_sigma(X, y, nstep=20, tol=1.e-5)
    print("estimated sigma", sigma)
    y /= sigma

    np.random.seed(0)
    lasso = randomized_lasso_trial(X,
                                   y,
                                   1.)

    adjusted_intervals = sigma * lasso[0]
    unadjusted_intervals = sigma * lasso[1]
    selective_mean = sigma * lasso[2]
    unadjusted_mean = sigma * lasso[3]
    ad_length = sigma * lasso[4]
    unad_length = sigma * lasso[5]

    print("unadjusted results", unadjusted_intervals, unadjusted_mean)
    print("adjusted results", adjusted_intervals, selective_mean)
    print("adjusted vs unadjusted lengths", ad_length, unad_length)