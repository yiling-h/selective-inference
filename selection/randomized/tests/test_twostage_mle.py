from selection.tests.instance import gaussian_instance

import numpy as np

from selection.randomized.lasso import full_targets, selected_targets
from selection.randomized.marginal_slope_twostage import marginal_screening, slope
from selection.randomized.randomization import randomization
from selection.randomized.query import twostage_selective_MLE

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

def test_marginal_slope(n=4000, p=2000, signal_fac=2., s=50, sigma=3., rho=0.20, randomizer_scale= np.sqrt(1.),
                        target = "selected"):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))
    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      s=s,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    sigma_ = np.sqrt(np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p))
    Y /= sigma_

    score = X.T.dot(Y)
    omega = randomization.isotropic_gaussian((p,), randomizer_scale * sigma_).sample()
    W = X.T.dot(X)
    marginal_select = marginal_screening.type1(score,
                                               W * sigma_ ** 2,
                                               0.1,
                                               randomizer_scale * sigma_,
                                               useC=True,
                                               perturb=omega)

    boundary, cond_mean_1, cond_cov_1, affine_con_1, logdens_linear_1, initial_soln_1 = marginal_select.fit()
    nonzero = boundary != 0

    print("selected", nonzero.sum())

    _, _, cov_target_score_1, _ = marginal_select.multivariate_targets(nonzero)

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
    print("final dimensions", nonzero_slope.sum())

    (observed_target,
     cov_target,
     cov_target_score_2,
     alternatives) = selected_targets(conv.loglike,
                                      conv._W,
                                      nonzero_slope,
                                      dispersion=1.)

    beta_target = np.linalg.pinv(X_tilde[:, nonzero_slope]).dot(X_tilde.dot(beta[nonzero])) / sigma_

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
                                                                affine_con_1.linear_part,
                                                                affine_con_1.offset,
                                                                affine_con_2.offset,
                                                                solve_args={'tol': 1.e-12},
                                                                level=0.9)

test_marginal_slope()


