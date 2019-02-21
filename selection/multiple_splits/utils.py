import numpy as np
from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def sim_xy(n, p, nval, alpha =3., rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    source('~/Research/Carving_causal_inference/simulation.R')
    sim_xy = sim.regression
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, alpha, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
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

def glmnet_lasso_cv1se(X, y):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate.1se = estimate.1se, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    estimate_1se = np.array(lambda_R(r_X, r_y).rx2('estimate.1se'))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.1se')))
    return estimate_1se, lam_1se


def glmnet_lasso_cvmin(X, y):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate.min = estimate.min,lam.min = fit.cv$lambda.min))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    estimate_min = np.array(lambda_R(r_X, r_y).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.min')))
    return estimate_min, lam_min