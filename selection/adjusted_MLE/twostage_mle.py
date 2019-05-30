import numpy as np
from selection.randomized.marginal_slope_twostage import marginal_screening, slope
from selection.randomized.randomization import randomization
from selection.randomized.query import twostage_selective_MLE
from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets

def coverage(intervals, pval, target, truth):
    pval_alt = (pval[truth != 0]) < 0.1
    if pval_alt.sum() > 0:
        avg_power = np.mean(pval_alt)
    else:
        avg_power = 0.
    return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power


try:
    from rpy2.robjects.packages import importr
    from rpy2 import robjects
    import rpy2.robjects.numpy2ri
    rpy_loaded = True
except ImportError:
    rpy_loaded = False

if rpy_loaded:

    def slope_R(X, Y, W = None, normalize = True, choice_weights = "gaussian", sigma = None):
        rpy2.robjects.numpy2ri.activate()
        robjects.r('''
        library('SLOPE')
        slope = function(X, Y, W , normalize, choice_weights, sigma, fdr = NA){
          if(is.na(sigma)){
          sigma=NULL} else{
          sigma = as.matrix(sigma)[1,1]}
          if(is.na(fdr)){
          fdr = 0.1 }
          if(normalize=="TRUE"){
           normalize = TRUE} else{
           normalize = FALSE}
          if(is.na(W))
          {
            if(choice_weights == "gaussian"){
            lambda = "gaussian"} else{
            lambda = "bhq"}
            result = SLOPE(X, Y, fdr = fdr, lambda = lambda, normalize = normalize, sigma = sigma)
           } else{
            result = SLOPE(X, Y, fdr = fdr, lambda = W, normalize = normalize, sigma = sigma)
          }
          return(list(beta = result$beta, E = result$selected, lambda_seq = result$lambda, sigma = result$sigma))
        }''')

        r_slope = robjects.globalenv['slope']

        n, p = X.shape
        r_X = robjects.r.matrix(X, nrow=n, ncol=p)
        r_Y = robjects.r.matrix(Y, nrow=n, ncol=1)

        if normalize is True:
            r_normalize = robjects.StrVector('True')
        else:
            r_normalize = robjects.StrVector('False')

        if W is None:
            r_W = robjects.NA_Logical
            if choice_weights is "gaussian":
                r_choice_weights  = robjects.StrVector('gaussian')
            elif choice_weights is "bhq":
                r_choice_weights = robjects.StrVector('bhq')
        else:
            r_W = robjects.r.matrix(W, nrow=p, ncol=1)

        if sigma is None:
            r_sigma = robjects.NA_Logical
        else:
            r_sigma = robjects.r.matrix(sigma, nrow=1, ncol=1)

        result = r_slope(r_X, r_Y, r_W, r_normalize, r_choice_weights, r_sigma)

        result = np.asarray(result.rx2('beta')), np.asarray(result.rx2('E')), \
            np.asarray(result.rx2('lambda_seq')), np.asscalar(np.array(result.rx2('sigma')))
        rpy2.robjects.numpy2ri.deactivate()

        return result

def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
    sim_xy = sim.xy
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

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
                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate = estimate, estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)
    r_lam = robjects.r.matrix(lambda_val, nrow=1, ncol=1)

    estimate = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate'))
    estimate_1se = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y, r_lam).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y, r_lam).rx2('lam.1se')))
    return estimate, estimate_1se, estimate_min, lam_min, lam_1se


def compare_twostage_mle(n=3000, p=1000, nval=3000, rho=0.35, s=35, beta_type=1, snr=0.20,
                         randomizer_scale=np.sqrt(0.50), full_dispersion=True):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    X -= X.mean(0)[None, :]
    scaling = X.std(0)[None, :] * np.sqrt(n)
    X /= scaling
    y = y - y.mean()

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    Y = y / sigma_

    score = X.T.dot(Y)
    omega = randomization.isotropic_gaussian((p,), randomizer_scale * sigma_).sample()
    W = X.T.dot(X)
    marginal_select = marginal_screening.type1(score,
                                               W,
                                               0.1,
                                               randomizer_scale,
                                               useC=True,
                                               perturb=omega)

    boundary, cond_mean_1, cond_cov_1, affine_con_1, logdens_linear_1, initial_soln_1 = marginal_select.fit()
    nonzero = boundary != 0
    first_selected = np.asarray([t for t in range(p) if nonzero[t]])

    X_tilde = X[:, nonzero]

    r_beta, r_E, r_lambda_seq, r_sigma = slope_R(X_tilde,
                                                 Y,
                                                 W=None,
                                                 normalize=True,
                                                 choice_weights="gaussian",  # put gaussian
                                                 sigma=1.)

    conv = slope.gaussian(X_tilde,
                          Y,
                          r_sigma * r_lambda_seq,
                          sigma=1.,
                          randomizer_scale=randomizer_scale * 1.)

    signs, cond_mean_2, cond_cov_2, affine_con_2, logdens_linear_2, initial_soln_2 = conv.fit()
    nonzero_slope = signs != 0
    second_selected = np.asarray([s for s in range(nonzero.sum()) if nonzero_slope[s]])

    stdev = np.sqrt(np.diag(X.T.dot(X)))
    boundary_nonrand = (score > stdev * ndist.ppf(1. - 0.10 / 2.))
    nonzero_nonrand = boundary_nonrand != 0
    first_selected_nonrand = np.asarray([z for z in range(p) if nonzero[z]])

    X_tilde_nonrand = X[:, nonzero_nonrand]

    r_beta_nonrand, r_E_nonrand, _, _ = slope_R(X_tilde_nonrand,
                                                Y,
                                                W=None,
                                                normalize=True,
                                                choice_weights="gaussian",  # put gaussian
                                                sigma=1.)

    nonzero_slope_nonrand = (r_beta_nonrand != 0)
    second_selected_nonrand = np.asarray([w for w in range(nonzero_nonrand.sum()) if nonzero_slope_nonrand[w]])

    print("compare dimensions- ms ", nonzero.sum(), nonzero_nonrand.sum())
    print("compare dimensions- slope ", nonzero_slope.sum(), nonzero_slope_nonrand.sum())

    nreport = 0.
    nreport_nonrand = 0.
    if nonzero_slope.sum()>0:
        _, _, cov_target_score_1, _ = marginal_select.multivariate_targets(first_selected[second_selected])

        (observed_target,
         cov_target,
         cov_target_score_2,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero_slope,
                                          dispersion=1.)

        beta_target = np.sqrt(n) * np.linalg.pinv(X_tilde[:, nonzero_slope]).dot(X_tilde.dot(beta[nonzero])) / sigma_

        estimate, _, _, pval, intervals, _ = twostage_selective_MLE(observed_target,
                                                                    cov_target,
                                                                    cov_target_score_1,
                                                                    cov_target_score_2,
                                                                    initial_soln_1,
                                                                    initial_soln_2,
                                                                    cond_mean_1,
                                                                    cond_mean_2,
                                                                    cond_cov_1,
                                                                    cond_cov_2,
                                                                    logdens_linear_1,
                                                                    logdens_linear_2,
                                                                    affine_con_1.linear_part,
                                                                    affine_con_2.linear_part,
                                                                    affine_con_1.offset,
                                                                    affine_con_2.offset,
                                                                    solve_args={'tol': 1.e-12},
                                                                    level=0.9)

        pval_alt = (pval[beta[first_selected[second_selected]] != 0]) < 0.1
        if pval_alt.sum() > 0:
            power_adjusted = np.mean(pval_alt)
        else:
            power_adjusted = 0.
        fdr_adjusted = ((pval[beta[first_selected[second_selected]] == 0]) < 0.1).sum()/float((pval<0.1).sum())

        coverage_adjusted = np.mean((beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1]))
        length_adjusted = sigma_* np.mean(intervals[:, 1] - intervals[:, 0])/np.sqrt(n)

        post_sel_OLS = np.linalg.pinv(X_tilde[:, nonzero_slope]).dot(Y)
        naive_sd = np.sqrt(np.diag((np.linalg.inv(X_tilde[:, nonzero_slope].T.dot(X_tilde[:, nonzero_slope])))))
        intervals_naive = np.vstack([post_sel_OLS - 1.65 * naive_sd,
                                     post_sel_OLS + 1.65 * naive_sd]).T
        coverage_naive = np.mean((beta_target > intervals_naive[:, 0]) * (beta_target < intervals_naive[:, 1]))
        length_naive = sigma_* np.mean(intervals_naive[:, 1] - intervals_naive[:, 0])/np.sqrt(n)

    else:
        nreport += 1
        coverage_adjusted, length_adjusted, power_adjusted, fdr_adjusted, coverage_naive, length_naive = [0., 0., 0., 0., 0., 0.]

    if nonzero_slope_nonrand.sum()>0:
        beta_target_nonrand = np.sqrt(n) * np.linalg.pinv(X_tilde_nonrand[:, nonzero_slope_nonrand]).dot(X_tilde_nonrand.dot(beta[nonzero_nonrand])) / sigma_
        post_sel_OLS_nonrand = np.linalg.pinv(X_tilde_nonrand[:, nonzero_slope_nonrand]).dot(Y)
        naive_sd_nonrand = np.sqrt(np.diag((np.linalg.inv(X_tilde_nonrand[:, nonzero_slope_nonrand].T.dot(X_tilde_nonrand[:, nonzero_slope_nonrand])))))
        intervals_naive_nonrand = np.vstack([post_sel_OLS_nonrand - 1.65 * naive_sd_nonrand,
                                             post_sel_OLS_nonrand + 1.65 * naive_sd_nonrand]).T
        coverage_naive_nonrand = np.mean((beta_target_nonrand > intervals_naive_nonrand[:, 0]) * (beta_target_nonrand < intervals_naive_nonrand[:, 1]))
        length_naive_nonrand = sigma_ * np.mean(intervals_naive_nonrand[:, 1] - intervals_naive_nonrand[:, 0])/np.sqrt(n)
        pval_nonrand = 2 * (1.-ndist.cdf(np.abs(post_sel_OLS_nonrand) / naive_sd_nonrand))

        pval_alt_nonrand = (pval_nonrand[beta[first_selected_nonrand[second_selected_nonrand]] != 0]) < 0.1
        if pval_alt_nonrand.sum() > 0:
            power_nonrand = np.mean(pval_alt_nonrand)
        else:
            power_nonrand = 0.
        fdr_nonrand = ((pval_nonrand[beta[first_selected_nonrand[second_selected_nonrand]] == 0]) < 0.1).sum() / float((pval_nonrand < 0.1).sum())
    else:
        nreport_nonrand += 1
        coverage_naive_nonrand, length_naive_nonrand, power__nonrand, fdr__nonrand = [0., 0., 0., 0.]

    MLE_inf = np.vstack((coverage_adjusted, length_adjusted, power_adjusted, fdr_adjusted, nonzero.sum(), nonzero_slope.sum()))
    #Naive_rand_inf = np.vstack((coverage_naive, length_naive, 0., 0.))
    Naive_inf = np.vstack((coverage_naive_nonrand, length_naive_nonrand, power_nonrand, fdr_nonrand, nonzero_nonrand.sum(), nonzero_slope_nonrand.sum()))
    print("inf", MLE_inf, Naive_inf)

    return np.vstack((MLE_inf, Naive_inf, nreport, nreport_nonrand))

#compare_twostage_mle()

def multiple_runs_lasso(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20,
                         randomizer_scale=np.sqrt(0.50), full_dispersion=True):


    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    glm_LASSO_theory, glm_LASSO_1se, glm_LASSO_min, lam_min, lam_1se = glmnet_lasso(X, y, lam_theory / float(n))

    active_LASSO_1 = (glm_LASSO_theory != 0)
    active_LASSO_2 = (glm_LASSO_1se != 0)
    active_LASSO = np.logical_or(active_LASSO_1, active_LASSO_2)
    nreport_nonrand = 0.
    if active_LASSO.sum()>0:
        target_nonrandomized = np.linalg.pinv(X[:, active_LASSO]).dot(X.dot(beta))
        post_LASSO_OLS = np.linalg.pinv(X[:, active_LASSO]).dot(y)

        naive_sd = sigma_ * np.sqrt(np.diag((np.linalg.inv(X[:, active_LASSO].T.dot(X[:, active_LASSO])))))
        naive_intervals = np.vstack([post_LASSO_OLS - 1.65 * naive_sd,
                                     post_LASSO_OLS + 1.65 * naive_sd]).T
        naive_pval = 2 * (1.-ndist.cdf(np.abs(post_LASSO_OLS)/ naive_sd))
        cov_naive, power_naive = coverage(naive_intervals, naive_pval, target_nonrandomized, beta[active_LASSO])
        length_naive = np.mean(naive_intervals[:, 1] - naive_intervals[:, 0])
        fdr_naive = ((naive_pval[beta[active_LASSO] == 0]) < 0.1).sum() / float((naive_pval < 0.1).sum())
    else:
        nreport_nonrand +=1.
        cov_naive, power_naive, length_naive, fdr_naive = [0.,0., 0.,0.]

    randomized_lasso_1 = lasso.gaussian(X,
                                        y,
                                        feature_weights=lam_theory * np.ones(p),
                                        randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs_1 = randomized_lasso_1.fit()
    nonzero_1 = signs_1 != 0

    randomized_lasso_2 = lasso.gaussian(X,
                                        y,
                                        feature_weights=n * lam_1se * np.ones(p),
                                        randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs_2 = randomized_lasso_2.fit()
    nonzero_2 = signs_2 != 0

    signs = np.logical_or(signs_1, signs_2)
    nonzero = signs!=0
    print("check", nonzero_1.sum(), nonzero_2.sum(), nonzero.sum(), active_LASSO.sum())
    nreport = 0.
    if nonzero.sum() > 0:
        target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
        observed_target = np.linalg.pinv(X[:, nonzero]).dot(y)
        (_,
         _,
         cov_target_score_1,
         alternatives_1) = selected_targets(randomized_lasso_1.loglike,
                                            randomized_lasso_1._W,
                                            nonzero,
                                            dispersion=dispersion)

        (_,
         cov_target,
         cov_target_score_2,
         alternatives_2) = selected_targets(randomized_lasso_2.loglike,
                                            randomized_lasso_2._W,
                                            nonzero,
                                            dispersion=dispersion)


        estimate, _, _, pval, intervals, _ = twostage_selective_MLE(observed_target,
                                                                    cov_target,
                                                                    cov_target_score_1,
                                                                    cov_target_score_2,
                                                                    randomized_lasso_1.observed_opt_state,
                                                                    randomized_lasso_2.observed_opt_state,
                                                                    randomized_lasso_1.cond_mean,
                                                                    randomized_lasso_2.cond_mean,
                                                                    randomized_lasso_1.cond_cov,
                                                                    randomized_lasso_2.cond_cov,
                                                                    randomized_lasso_1.logdens_linear,
                                                                    randomized_lasso_2.logdens_linear,
                                                                    randomized_lasso_1.con_linear,
                                                                    randomized_lasso_2.con_linear,
                                                                    randomized_lasso_1.con_offset,
                                                                    randomized_lasso_2.con_offset,
                                                                    solve_args={'tol': 1.e-12},
                                                                    level=0.9)

        coverage_adjusted, power_adjusted = coverage(intervals, pval, target_randomized, beta[nonzero])
        length_adjusted = np.mean(intervals[:, 1] - intervals[:, 0])
        fdr_adjusted = ((pval[beta[nonzero] == 0]) < 0.1).sum() / float((pval < 0.1).sum())

    else:
        nreport +=1
        coverage_adjusted, length_adjusted, power_adjusted, fdr_adjusted = [0., 0., 0., 0.]

    MLE_inf = np.vstack((coverage_adjusted, length_adjusted, power_adjusted, fdr_adjusted, nonzero.sum()))
    Naive_inf = np.vstack((cov_naive, length_naive, power_naive, fdr_naive, active_LASSO.sum()))

    print MLE_inf, Naive_inf
    return np.vstack((MLE_inf, Naive_inf, nreport, nreport_nonrand))


