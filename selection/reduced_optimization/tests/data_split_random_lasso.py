from __future__ import print_function
import numpy as np
import regreg.api as rr
from selection.tests.instance import logistic_instance, gaussian_instance
from selection.reduced_optimization.par_random_lasso_reduced import selection_probability_random_lasso, sel_inf_random_lasso
from selection.reduced_optimization.estimator import M_estimator_approx

def generate_data_random(n, p, sigma=1., rho=0., scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.9:
            beta_true[i] = np.random.laplace(loc=0., scale=0.1)
        else:
            beta_true[i] = np.random.laplace(loc=0., scale=1.)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta * sigma, sigma

def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma,
                           loss ='gaussian',
                           randomizer='gaussian',
                           estimation='parametric'):

    from selection.api import randomization

    n, p = X.shape
    sample = np.random.choice(n, int((n)/2.), replace=False)
    sample_indices = np.zeros(n, np.bool)
    sample_indices[sample] = 1
    print("sample", n, sample_indices.sum())
    X_select = X[sample_indices, :]
    y_select = y[sample_indices]
    X_inf = X[~sample_indices, :]
    y_inf = y[~sample_indices]

    loss = rr.glm.gaussian(X_select, y_select)

    epsilon = 1. / np.sqrt((n)/2.)

    lam = 1. * np.mean(np.fabs(np.dot(X_select.T, np.random.standard_normal((int((n)/2.), 2000)))).max(0)) * sigma
    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),weights=dict(zip(np.arange(p), W)), lagrange=1.)
    randomization = randomization.isotropic_gaussian((p,), scale=1.)

    M_est = M_estimator_approx(loss, epsilon, penalty, randomization, randomizer, estimation)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    prior_variance = 10000.
    noise_variance = sigma ** 2
    projection_active = X_inf[:, active].dot(np.linalg.inv(X_inf[:, active].T.dot(X_inf[:, active])))
    M_1 = prior_variance * (X_inf.dot(X_inf.T)) + noise_variance * np.identity(int(n/2.))
    M_2 = prior_variance * ((X_inf.dot(X_inf.T)).dot(projection_active))
    M_3 = prior_variance * (projection_active.T.dot(X_inf.dot(X_inf.T)).dot(projection_active))
    post_mean = M_2.T.dot(np.linalg.inv(M_1)).dot(y_inf)

    print("observed data", post_mean)

    post_var = M_3 - M_2.T.dot(np.linalg.inv(M_1)).dot(M_2)

    unadjusted_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                                      post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

    coverage_unad = np.zeros(nactive)
    unad_length = np.zeros(nactive)

    true_val = np.zeros(nactive)

    for l in range(nactive):
        if (unadjusted_intervals[0, l] <= true_val[l]) and (true_val[l] <= unadjusted_intervals[1, l]):
            coverage_unad[l] += 1
        unad_length[l] = unadjusted_intervals[1, l] - unadjusted_intervals[0, l]

    naive_cov = coverage_unad.sum() / nactive
    unad_len = unad_length.sum() / nactive
    bayes_risk_unad = np.power(post_mean - true_val, 2.).sum() / nactive

    return np.vstack([naive_cov, unad_len, bayes_risk_unad])


if __name__ == "__main__":
    ### set parameters
    n = 1000
    p = 100
    s = 0
    snr = 0.


    niter = 50
    unad_cov = 0.
    unad_len = 0.
    unad_risk = 0.

    for i in range(niter):

         ### GENERATE X, Y BASED ON SEED
         #np.random.seed(i+100)  # ensures different X and y
         #X, y, beta , nonzero, sigma = gaussian_instance(n=n, p=p, s=s, sigma=1., rho=0., snr=snr)
         np.random.seed(i)
         X, y, beta, sigma = generate_data_random(n=n, p=p)

         ### RUN LASSO AND TEST
         lasso = randomized_lasso_trial(X,
                                        y,
                                        beta,
                                        sigma)

         if lasso is not None:
             unad_cov += lasso[0,0]
             unad_len += lasso[1,0]
             unad_risk += lasso[2,0]

             print("\n")
             print("iteration completed", i)
             print("\n")
             print("adjusted and unadjusted coverage", unad_cov)
             print("adjusted and unadjusted lengths",unad_len)
             print("adjusted and unadjusted risks", unad_risk)


    print("unadjusted coverage",unad_cov/50.)
    print("unadjusted lengths", unad_len/50.)
    print("unadjusted risks", unad_risk/50.)