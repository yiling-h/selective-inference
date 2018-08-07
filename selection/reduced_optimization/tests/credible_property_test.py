from __future__ import print_function
import numpy as np, sys
import regreg.api as rr

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y,lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.matrix(lambda)[1,1]
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lam, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)
    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    return estimate

def generate_data_random(n, p, sigma=1., rho=0.2, scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))

    prior_mean = 0.
    prior_sd = 1.
    beta = np.random.normal(prior_mean, prior_sd, p)

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta * sigma, sigma, prior_mean, prior_sd

def Bayesian_inference(seedn):

    np.random.seed(seedn)
    X, y, beta, sigma, prior_mean, prior_sd = generate_data_random(n=500, p=100)

    #prior_var = prior_sd ** 2
    prior_mean = -1.
    prior_var = 10 ** 2.
    n, p = X.shape
    true_mean = X.dot(beta)
    X_scaled = np.sqrt(n) * X
    lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma

    glm_LASSO = glmnet_lasso(X_scaled, y, lam / np.sqrt(n))
    active_LASSO = (glm_LASSO != 0)
    print("nactive selected by LASSO", active_LASSO.sum())
    projection_target = np.linalg.pinv(X[:, active_LASSO]).dot(true_mean)

    marginal_prec = np.linalg.inv(prior_var*(X.dot(X.T))+ (sigma**2.)*np.identity(n))
    posterior_mean = prior_mean*np.ones(p) + prior_var*(X.T.dot(marginal_prec).dot(y-X.dot(prior_mean*np.ones(p))))
    posterior_var = prior_var* (np.identity(p) - (prior_var*X.T.dot(marginal_prec).dot(X)))

    projection_op = np.linalg.pinv(X[:, active_LASSO]).dot(X)
    post_mean = projection_op.dot(posterior_mean)
    post_var = projection_op.dot(posterior_var).dot(projection_op.T)

    credible_intervals = np.vstack([post_mean - 1.65 * (np.sqrt(post_var.diagonal())),
                           post_mean + 1.65 * (np.sqrt(post_var.diagonal()))])

    coverage = np.mean((projection_target > credible_intervals[0, :]) * (projection_target < credible_intervals[1, :]))

    return coverage

def main(nsim = 100):

    credible_coverage = 0.
    for i in range(nsim):
        credible_coverage += Bayesian_inference(i)
        print("iteration completed", i + 1)
        print("credible coverage so far ", credible_coverage / float(i + 1))

main()


