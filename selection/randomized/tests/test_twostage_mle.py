from selection.tests.instance import gaussian_instance

import numpy as np

from selection.randomized.lasso import full_targets, selected_targets
from selection.randomized.marginal_slope_twostage import marginal_screening, slope
from selection.randomized.randomization import randomization
from selection.randomized.query import twostage_selective_MLE
from scipy.stats import norm as ndist

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

def test_marginal_slope(n=3000, p=1000, signal_fac=1.5, s=30, sigma=1., rho=0.20, randomizer_scale= np.sqrt(1.),
                        split_proportion= 0.50, target = "selected"):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))
    X, y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      s=s,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    sigma_ = np.sqrt(np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p))
    #sigma_ = np.std(y)/np.sqrt(2)
    #sigma_ = 1.
    Y = y/sigma_

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

    subsample_size = int(split_proportion * n)
    sel_idx = np.zeros(n, np.bool)
    sel_idx[:subsample_size] = 1
    np.random.shuffle(sel_idx)
    inf_idx = ~sel_idx

    Y_inf = Y[inf_idx]
    X_inf = X[inf_idx, :]
    #_sigma_ = np.sqrt(np.linalg.norm(Y_inf - X_inf.dot(np.linalg.pinv(X_inf).dot(Y_inf))) ** 2 / (n - p))
    Y_sel = Y[sel_idx]
    X_sel = X[sel_idx, :]
    #Y_inf /= _sigma_

    score_split = X_sel.T.dot(Y_sel)
    stdev_split = np.sqrt(np.diag(X_sel.T.dot(X_sel)))
    threshold_split = stdev_split * ndist.ppf(1. - 0.1/ 2.)
    boundary_split = np.fabs(score_split) >= threshold_split
    nonzero_split = boundary_split != 0
    first_selected_split = np.asarray([u for u in range(p) if nonzero_split[u]])

    X_tilde_sel = X_sel[:, nonzero_split]
    r_beta_split, r_E_split, r_lambda_seq_split, r_sigma_split = slope_R(X_tilde_sel,
                                                                         Y_sel,
                                                                         W=None,
                                                                         normalize=True,
                                                                         choice_weights="gaussian",
                                                                         sigma=1.)

    nonzero_slope_split = (r_beta_split != 0)
    second_selected_split = np.asarray([r for r in range(nonzero_split.sum()) if nonzero_slope_split[r]])

    print("compare dimensions- ms ", nonzero.sum(), nonzero_split.sum())
    print("compare dimensions- slope ", nonzero_slope.sum(), nonzero_slope_split.sum())

    beta_target_split = np.linalg.pinv(X_inf[:, first_selected_split[second_selected_split]]).dot(X_inf[:, first_selected_split].dot(beta[nonzero_split]))/ sigma_
    post_split_OLS = np.linalg.pinv(X_inf[:, first_selected_split[second_selected_split]]).dot(Y_inf)
    naive_split_sd = np.sqrt(np.diag((np.linalg.inv(X_inf[:, first_selected_split[second_selected_split]].T.dot(X_inf[:, first_selected_split[second_selected_split]])))))
    intervals_split = np.vstack([post_split_OLS - 1.65 * naive_split_sd,
                                 post_split_OLS + 1.65 * naive_split_sd]).T
    coverage_split = (beta_target_split > intervals_split[:, 0]) * (beta_target_split < intervals_split[:, 1])
    length_split = intervals_split[:, 1] - intervals_split[:, 0]
    pval_split = 2 *(1.-ndist.cdf(np.abs(post_split_OLS) / naive_split_sd))

    pval_alt_split = (pval_split[beta[first_selected_split[second_selected_split]] != 0]) < 0.1
    if pval_alt_split.sum() > 0:
        power_split = np.mean(pval_alt_split)
    else:
        power_split = 0.

    if target == "selected":
        _, _, cov_target_score_1, _ = marginal_select.multivariate_targets(first_selected[second_selected])

        (observed_target,
         cov_target,
         cov_target_score_2,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero_slope,
                                          dispersion=1.)

        beta_target = np.linalg.pinv(X_tilde[:, nonzero_slope]).dot(X_tilde.dot(beta[nonzero])) / sigma_

    elif target == "full":
        _, _, cov_target_score_1, _ = marginal_select.marginal_targets(first_selected[second_selected])

        (observed_target,
         cov_target,
         cov_target_score_2,
         alternatives) = full_targets(conv.loglike,
                                      conv._W,
                                      nonzero_slope,
                                      dispersion=1.)

        beta_target = beta[first_selected[second_selected]] / sigma_

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

    coverage_adjusted = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
    length_adjusted = intervals[:, 1] - intervals[:, 0]

    post_sel_OLS = np.linalg.pinv(X_tilde[:, nonzero_slope]).dot(Y)
    naive_sd = np.sqrt(np.diag((np.linalg.inv(X_tilde[:, nonzero_slope].T.dot(X_tilde[:, nonzero_slope])))))
    intervals_naive = np.vstack([post_sel_OLS - 1.65 * naive_sd,
                                 post_sel_OLS + 1.65 * naive_sd]).T
    coverage_naive = (beta_target > intervals_naive[:, 0]) * (beta_target < intervals_naive[:, 1])
    length_naive = intervals_naive[:, 1] - intervals_naive[:, 0]

    return coverage_adjusted, sigma_ * length_adjusted, power_adjusted, coverage_naive, sigma_ * length_naive, coverage_split, sigma_ * length_split, power_split


def main(nsim=100):
    cover_adjusted, length_adjusted, power_adjusted, cover_naive, length_naive, \
    cover_split, length_split, power_split = [], [], 0., [], [], [], [], 0.

    for i in range(nsim):
        results_ = test_marginal_slope()

        cover_adjusted.extend(results_[0])
        cover_naive.extend(results_[3])
        cover_split.extend(results_[5])
        length_adjusted.extend(results_[1])
        length_naive.extend(results_[4])
        length_split.extend(results_[6])
        power_split += results_[7]
        power_adjusted += results_[2]

        print('coverage and lengths', np.mean(cover_adjusted), np.mean(cover_split), np.mean(cover_naive), np.mean(length_adjusted),
        np.mean(length_split), np.mean(length_naive), power_adjusted/float(i+1), power_split/float(i+1))

main()


