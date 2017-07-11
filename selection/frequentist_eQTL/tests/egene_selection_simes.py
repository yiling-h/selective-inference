from __future__ import print_function
from scipy.stats import norm as normal
import numpy as np
import os
import sys
import regreg.api as rr
import statsmodels.api as sm

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

def estimate_sigma(X, y, nstep=30, tol=1.e-3):

    old_sigma = 0.5
    old_old_sigma = old_sigma
    n, p = X.shape

    for itercount in range(nstep):

        random_Z = np.zeros(p)
        sel = selection(X, y, random_Z, sigma=old_sigma)
        if sel is not None:
            lam, epsilon, active, betaE, cube, initial_soln = sel
            sys.stderr.write("active" + str(active.sum()) + "\n")
            if active.sum()<n-1:
                ols_fit = sm.OLS(y, X[:, active]).fit()
                new_sigma = np.linalg.norm(ols_fit.resid) / np.sqrt(n - active.sum() - 1)
            else:
                new_sigma = 0.75
        else:
            new_sigma = old_sigma/2.

        sys.stderr.write("est_sigma" + str(new_sigma) + str(old_sigma)+ "\n")
        if np.fabs(new_sigma - old_sigma) < tol :
            sigma = new_sigma
            break
        if np.fabs(new_sigma - old_old_sigma) < 0.001*tol :
            sigma = new_sigma
            break
        old_old_sigma = old_sigma
        old_sigma = new_sigma
        sigma = new_sigma

    return sigma

def simes_selection_egene(X,
                          y,
                          randomizer= 'gaussian',
                          noise_level = 1.,
                          randomization_scale=1.):

    n, p = X.shape
    sigma = noise_level

    T_stats = X.T.dot(y) / sigma

    if randomizer == 'gaussian':
        perturb = np.random.standard_normal(p)
        randomized_T_stats = T_stats + randomization_scale * perturb

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats),
                                                                        np.sqrt(1.+(randomization_scale**2))))))

    elif randomizer == 'none':

        randomized_T_stats = T_stats

        p_val_randomized = np.sort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)))))

        indices_order = np.argsort(2. * (1. - normal.cdf(np.true_divide(np.abs(randomized_T_stats), np.sqrt(1.)))))

    simes_p_randomized = np.min((p / (np.arange(p) + 1.)) * p_val_randomized)

    i_0 = np.argmin((p / (np.arange(p) + 1.)) * p_val_randomized)

    t_0 = indices_order[i_0]

    T_stats_active = T_stats[i_0]

    u_1 = ((i_0 + 1.) / p) * np.min(
        np.delete((p / (np.arange(p) + 1.)) * p_val_randomized, i_0))

    if i_0 > p - 2:
        u_2 = -1
    else:
        u_2 = p_val_randomized[i_0 + 1]

    return simes_p_randomized, i_0, t_0, np.sign(T_stats_active), u_1, u_2


# if __name__ == "__main__":
#
#     path = sys.argv[1]
#     outdir = sys.argv[2]
#     result = sys.argv[3]
#
#     outfile = os.path.join(outdir, "part1_simes_output_sigma_estimated_"+ str(result) + ".txt")
#
#     gene_file = path + "Genes.txt"
#
#     with open(gene_file) as g:
#         content = g.readlines()
#
#     content = [x.strip() for x in content]
#     sys.stderr.write("length" + str(len(content)) + "\n")
#
#     iter = int(len(content)/2.)
#     output = np.zeros((iter, 8))
#
#     for j in range(iter):
#
#         X = np.load(os.path.join(path + "X_" + str(content[j])) + ".npy")
#         n, p = X.shape
#         X -= X.mean(0)[None, :]
#         X /= (X.std(0)[None, :] * np.sqrt(n))
#
#         y = np.load(os.path.join(path + "y_" + str(content[j])) + ".npy")
#         y = y.reshape((y.shape[0],))
#
#         sigma = estimate_sigma(X, y, nstep=30, tol=1.e-3)
#         y /= sigma
#         sys.stderr.write("iteration completed" + str(j) + "\n")
#         sys.stderr.write("est sigma" + str(sigma) + "\n")
#         # run Simes
#         simes = simes_selection_egene(X, y, randomizer='gaussian')
#
#         output[j, 0] = p
#         output[j, 1] = sigma
#         output[j, 2:] = simes
#
#         #beta = np.load(os.path.join(path + "b_" + str(content[j])) + ".npy")
#
#     np.savetxt(outfile, output)

if __name__ == "__main__":

    path = sys.argv[1]
    outdir = sys.argv[2]
    result = sys.argv[3]

    outfile = os.path.join(outdir, "part2_simes_output_sigma_estimated_"+ str(result) + ".txt")

    gene_file = path + "Genes.txt"

    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]
    sys.stderr.write("length" + str(len(content)) + "\n")

    iter_0 = int(len(content)/2.)

    if len(content) % 2 == 0:
       iter = iter_0
    else:
       iter = iter_0 + 1

    output = np.zeros((iter, 8))
    for j in range(iter):

        k = j+ iter_0
        X = np.load(os.path.join(path + "X_" + str(content[k])) + ".npy")
        n, p = X.shape
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n))

        y = np.load(os.path.join(path + "y_" + str(content[k])) + ".npy")
        y = y.reshape((y.shape[0],))

        sigma = estimate_sigma(X, y, nstep=30, tol=1.e-3)
        y /= sigma
        sys.stderr.write("iteration completed" + str(j) + "\n")
        sys.stderr.write("est sigma" + str(sigma) + "\n")
        # run Simes
        simes = simes_selection_egene(X, y, randomizer='gaussian')

        output[j, 0] = p
        output[j, 1] = sigma
        output[j, 2:] = simes

        #beta = np.load(os.path.join(path + "b_" + str(content[j])) + ".npy")

    np.savetxt(outfile, output)
