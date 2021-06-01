import numpy as np
from scipy.stats import norm as ndist

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from ..lasso import lasso, full_targets, selected_targets, debiased_targets
from ...tests.instance import gaussian_instance


def glmnet_lasso(X, y):
    robjects.r('''
                library(glmnet)
                glmnet_LASSO = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)
                n = nrow(X)

                fit.cv = cv.glmnet(X, y, standardize=TRUE, intercept=FALSE, thresh=1.e-10)
                estimate.1se = coef(fit.cv, s='lambda.1se', exact=TRUE, x=X, y=y)[-1]
                estimate.min = coef(fit.cv, s='lambda.min', exact=TRUE, x=X, y=y)[-1]
                return(list(estimate.1se = estimate.1se, estimate.min = estimate.min, lam.min = fit.cv$lambda.min, lam.1se = fit.cv$lambda.1se))
                }''')

    lambda_R = robjects.globalenv['glmnet_LASSO']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    estimate_1se = np.array(lambda_R(r_X, r_y).rx2('estimate.1se'))
    estimate_min = np.array(lambda_R(r_X, r_y).rx2('estimate.min'))
    lam_min = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.min')))
    lam_1se = np.asscalar(np.array(lambda_R(r_X, r_y).rx2('lam.1se')))
    return estimate_1se, estimate_min, lam_min, lam_1se


def test_Full_targets(n=200,
                      p=1000, 
                      signal_fac=0.5,
                      s=5,
                      sigma=1.,
                      rho=0.4, 
                      randomizer_scale=0.5,
                      full_dispersion=False,
                      CV=False):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    while True:
        signal = np.sqrt(signal_fac * 2 * np.log(p))
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)

        if CV == False:
            eps = np.random.standard_normal((n, 2000)) * sigma_
            lam_theory = 1. * np.median(np.abs(X.T.dot(eps)).max(1))
            lam = lam_theory
        else:
            lam_1se = glmnet_lasso(X, Y)[3]
            lam = np.sqrt(n)*lam_1se

        conv = const(X,
                     Y,
                     lam * np.ones(p),
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        print("check ", nonzero.sum())

        if nonzero.sum() > 0:
            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = None

            if n>p:
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = full_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)
            else:
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = debiased_targets(conv.loglike,
                                                  conv._W,
                                                  nonzero,
                                                  penalty=conv.penalty,
                                                  dispersion=dispersion)

            result = conv.selective_MLE_inference(observed_target,
                                                  cov_target,
                                                  cov_target_score)[0]

            estimate = result['MLE']
            se = result['SE']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])

            coverage = (beta[nonzero] > intervals[:, 0]) * (beta[nonzero] < intervals[:, 1])
            pivot_ = ndist.cdf((estimate - beta[nonzero]) / se)
            pivot = 2 * np.minimum(pivot_, 1. - pivot_)

            return pivot, coverage, intervals


def test_Partial_targets(n=2000,
                         p=200,
                         signal_fac=1.,
                         s=5,
                         sigma=1,
                         rho=0.4,
                         randomizer_scale=1,
                         full_dispersion=True,
                         CV=False):
    """
    Compare to R randomized lasso
    """

    inst, const = gaussian_instance, lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    while True:
        X, Y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        sigma_ = np.std(Y)

        if CV == False:
            eps = np.random.standard_normal((n, 2000)) * sigma_
            lam_theory = 1. * np.median(np.abs(X.T.dot(eps)).max(1))
            lam = lam_theory
        else:
            lam_1se = glmnet_lasso(X, Y)[3]
            lam = np.sqrt(n)*lam_1se

        conv = const(X,
                     Y,
                     lam * np.ones(p),
                     randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0

        print("check ", nonzero.sum())

        if nonzero.sum() > 0:

            if full_dispersion:
                dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
            else:
                dispersion = None

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero, 
                                              dispersion=dispersion)

            result = conv.selective_MLE_inference(observed_target,
                                                  cov_target,
                                                  cov_target_score)[0]

            estimate = result['MLE']
            se = result['SE']
            intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
            
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])

            pivot_ = ndist.cdf((estimate - beta_target)/se)
            pivot = 2 * np.minimum(pivot_, 1. - pivot_)

            return pivot, coverage, intervals


def main(nsim=500, full=False, plot_ECDF=False):

    pivot, cover, length_int = [], [], []

    n, p, s = 500, 100, 5

    for i in range(nsim):
        if full:
            if n > p:
                full_dispersion = True
            else:
                full_dispersion = False
            p0, cover_, intervals = test_Full_targets(n=n, p=p, s=s, full_dispersion=full_dispersion, CV=True)
            avg_length = intervals[:, 1] - intervals[:, 0]
        else:
            full_dispersion = True
            p0, cover_, intervals = test_Partial_targets(n=n, p=p, s=s, full_dispersion=full_dispersion, CV=True)
            avg_length = intervals[:, 1] - intervals[:, 0]

        cover.extend(cover_)
        pivot.extend(p0)
        print("iteration", i, " coverage and lengths ", np.mean(cover), np.mean(avg_length))

    if plot_ECDF == True:
        import matplotlib as mpl
        mpl.use('tkagg')
        import matplotlib.pyplot as plt
        from statsmodels.distributions.empirical_distribution import ECDF

        plt.clf()

        ecdf_MLE = ECDF(np.asarray(pivot))

        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_MLE(grid), c='blue', linewidth=5)
        plt.plot(grid, grid, 'k--')
        plt.show()


if __name__ == "__main__":
    main(nsim=100, full=True, plot_ECDF=True)
