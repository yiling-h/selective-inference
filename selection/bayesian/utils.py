import numpy as np

from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

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

def glmnet_lasso(X, y, lambda_val):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y, lambda){
                y = as.matrix(y)
                X = as.matrix(X)
                lam = as.vector(lambda)
                n = nrow(X)
                fit = glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate = coef(fit, s=lambda, exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)

    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    return estimate

def log_likelihood(target_parameter,
                   observed_target,
                   cov_target,
                   cov_target_score,
                   feasible_point,
                   cond_mean,
                   cond_cov,
                   logdens_linear,
                   linear_part,
                   offset,
                   solve_args={'tol': 1.e-12}):

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)

    target_lin = - logdens_linear.dot(cov_target_score.T.dot(prec_target))
    target_offset = cond_mean - target_lin.dot(observed_target)

    prec_opt = np.linalg.inv(cond_cov)
    mean_opt = target_lin.dot(target_parameter)+target_offset
    conjugate_arg = prec_opt.dot(mean_opt)

    solver = solve_barrier_affine_C

    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             feasible_point,
                             linear_part,
                             offset,
                             **solve_args)

    reparam = target_parameter + cov_target.dot(target_lin.T.dot(prec_opt.dot(mean_opt - soln)))
    neg_normalizer = (target_parameter-reparam).T.dot(prec_target).dot(target_parameter-reparam)+ val + mean_opt.T.dot(prec_opt).dot(mean_opt)/2.

    L = target_lin.T.dot(prec_opt)
    jacobian = (np.identity(observed_target.shape[0])+ cov_target.dot(L).dot(target_lin)) - \
               cov_target.dot(L).dot(hess).dot(L.T)

    log_lik = -(observed_target-target_parameter).T.dot(prec_target).dot(observed_target-target_parameter)/2. + neg_normalizer \
              + np.log(np.linalg.det(jacobian))

    return log_lik

def coverage(intervals, target, truth):

    cov = (0. > intervals[:, 0]) * (0. < intervals[:, 1])
    pval_alt = (cov[truth != 0]) != 1
    if pval_alt.sum() > 0:
        avg_power = np.mean(pval_alt)
    else:
        avg_power = 0.
    return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power

def power_fdr(select_set, true_set):

    if select_set.shape[0] > 0 and true_set.shape[0] > 0:
        diff_set = np.fabs(np.subtract.outer(select_set, true_set))
        true_discoveries = (diff_set.min(0) == 0).sum()
        return true_discoveries

    else:
        return 0.


def discoveries_count_clusters(active_set, signal_clusters, false_clusters, clusters):

    true_discoveries = 0.
    false_discoveries = 0.
    discoveries = 0.
    for i in range(len(signal_clusters)):
        inter = np.intersect1d(active_set, signal_clusters[i])
        if inter.shape[0]>0:
            true_discoveries += 1
    for j in range(len(clusters)):
        inter = np.intersect1d(active_set, clusters[j])
        if inter.shape[0]>0:
            discoveries += 1
    for k in range(len(false_clusters)):
        inter = np.intersect1d(active_set, false_clusters[k])
        if inter.shape[0] > 0:
            false_discoveries += 1

    return true_discoveries, false_discoveries, discoveries

from scipy.stats.stats import pearsonr

def discoveries_count(active_set, true_signals, false_signals, X):

    true_discoveries = 0.
    false_discoveries = 0.
    discoveries = active_set.shape[0]
    for h in range(discoveries):
        corr_true = np.zeros(true_signals.shape[0])
        for i in range(true_signals.shape[0]):
            corr_true[i] = pearsonr(X[:, active_set[h]], X[:, true_signals[i]])[0]
        if np.any(corr_true >= 0.55):
            true_discoveries += 1.
        else:
            if np.min(np.fabs(false_signals-active_set[h])) == 0:
                false_discoveries += 1.

    return true_discoveries, false_discoveries, discoveries