import numpy as np
from random import seed

from selection.randomized.group_lasso import group_lasso
from selection.tests.instance import gaussian_group_instance, gaussian_instance

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm as ndist

p_temp = 25
def test_selected_targets(n=50,
                          p=p_temp,
                          signal_fac=0,
                          sgroup=1,
                          #s =10,
                          groups=np.arange(1).repeat(p_temp),
                          sigma=1.,
                          rho=0,
                          randomizer_scale=1,
                          weight_frac=.7):

    inst = gaussian_group_instance
    #inst = gaussian_instance
    const = group_lasso.gaussian
    signal = np.sqrt(signal_fac * 2 * np.log(p))

    # X, Y, beta = inst(n=n,
    #                   p=p,
    #                   signal=signal,
    #                   s=s,
    #                   equicorrelated=False,
    #                   rho=rho,
    #                   sigma=sigma,
    #                   random_signs=True)[:3]

    X, Y, beta = inst(n=n,
                      p=p,
                      signal=signal,
                      sgroup=sgroup,
                      groups=groups,
                      equicorrelated=False,
                      rho=rho,
                      sigma=sigma,
                      random_signs=True)[:3]

    n, p = X.shape

    sigma_ = np.std(Y)
    omega = randomization.isotropic_gaussian((p,), randomizer_scale * sigma_).sample()

    weights = dict([(i, weight_frac * sigma_ * 2 * np.sqrt(2)) for i in np.unique(groups)])
    conv = const(X, Y, groups, weights, randomizer_scale=randomizer_scale * sigma_,perturb=omega)
    signs,_ = conv.fit()
    nonzero = signs != 0
    print("check dimensions of selected set ", nonzero.sum())

    if nonzero.sum() > 0:
        if n>p:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        estimate, observed_info_mean, _, pval, intervals, _ = conv.selective_MLE(dispersion=dispersion,
                                                                                 useJacobian=True,
                                                                                 level=0.9)

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        pivot_MLE = ndist.cdf((estimate - beta_target)/np.sqrt(np.diag(observed_info_mean)))

        coverage = (beta_target > intervals[:, 0]) * (beta_target <
                                                      intervals[:, 1])
        p0 = pval[beta[nonzero] == 0]
        pa = pval[beta[nonzero] != 0]
    else:
        p0 = []
        pa = []
        coverage = []
        intervals = []
        pivot_MLE = []
        estimate = np.zeros(p)
        observed_info_mean = np.zeros((p,p))
        
    conv = const(X, Y, groups, weights, randomizer_scale=randomizer_scale * sigma_,perturb=omega)
    signs,_ = conv.fit()
    nonzero = signs != 0
    print("check dimensions of selected set ", nonzero.sum())

    if nonzero.sum() > 0:
        if n>p:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2

        estimate_nj, observed_info_mean_nj, _, pval_nj, intervals_nj, _ = conv.selective_MLE(dispersion=dispersion,
                                                                                 useJacobian=False,
                                                                                 level=0.9)

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        pivot_MLE_nj = ndist.cdf((estimate_nj - beta_target)/np.sqrt(np.diag(observed_info_mean_nj)))

        coverage_nj = (beta_target > intervals_nj[:, 0]) * (beta_target <
                                                      intervals_nj[:, 1])
        p0_nj = pval_nj[beta[nonzero] == 0]
        pa_nj = pval_nj[beta[nonzero] != 0]
    else:
        p0_nj = []
        pa_nj = []
        coverage_nj = []
        intervals_nj = []
        pivot_MLE_nj = []
        estimate_nj = np.zeros(p)
        observed_info_mean_nj = np.zeros((p,p))
        
    return p0,pa,coverage,intervals,pivot_MLE,estimate,observed_info_mean,p0_nj,pa_nj,coverage_nj,intervals_nj,pivot_MLE_nj,estimate_nj,observed_info_mean_nj,


def main(nsim=500,p=2):
    P0, PA, cover, pivot, avg_length = [], [], [], [], []
    P0_nj, PA_nj, cover_nj, pivot_nj, avg_length_nj = [], [], [], [], []

    beta_hat = np.zeros((nsim,p))
    Sigma_hat = np.zeros((nsim,p,p))
    beta_hat_nj = np.zeros((nsim,p))
    Sigma_hat_nj = np.zeros((nsim,p,p))
    # set parameters through defaults in function definition
    #n, p, sgroup = 200, 50, 1
    nselect = 0
    nselect_nj = 0
    for i in range(nsim):
        p0, pA, cover_, intervals, pivot_, beta_hat_, Sigma_hat_,p0_nj, pA_nj, cover_nj_, intervals_nj, pivot_nj_, beta_hat_nj_, Sigma_hat_nj_ = test_selected_targets()
        if len(intervals)>0:
            nselect += 1
            avg_length_ = intervals[:, 1] - intervals[:, 0]
        else:
            avg_length_ = []

        cover.extend(cover_)
        pivot.extend(pivot_)
        avg_length.extend(avg_length_)
        P0.extend(p0)
        PA.extend(pA)
        # store estimate and covariance
        beta_hat[i,:] = beta_hat_
        Sigma_hat[i,:,:] = Sigma_hat_
        
        if len(intervals)>0:
            print(np.mean(cover), np.mean(avg_length),
                  'coverage + length so far with Jacobian')
            
        if len(intervals_nj)>0:
            nselect_nj += 1
            avg_length_nj_ = intervals_nj[:, 1] - intervals_nj[:, 0]
        else:
            avg_length_nj_ = []

        cover_nj.extend(cover_nj_)
        pivot_nj.extend(pivot_nj_)
        avg_length_nj.extend(avg_length_nj_)
        P0_nj.extend(p0_nj)
        PA_nj.extend(pA_nj)
        # store estimate and covariance
        beta_hat_nj[i,:] = beta_hat_nj_
        Sigma_hat_nj[i,:,:] = Sigma_hat_nj_
        
        if len(intervals_nj)>0:
            print(np.mean(cover_nj), np.mean(avg_length_nj),
                  'coverage + length so far without Jacobian')

    plt.clf()
    ecdf_MLE = ECDF(np.asarray(pivot))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()
    
    plt.clf()
    ecdf_MLE = ECDF(np.asarray(pivot_nj))
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, ecdf_MLE(grid), c='red', marker='^')
    plt.plot(grid, grid, 'k--')
    plt.show()
    
    return nselect/nsim, np.mean(cover), np.mean(avg_length), beta_hat, Sigma_hat, nselect_nj/nsim, np.mean(cover_nj), np.mean(avg_length_nj), beta_hat_nj, Sigma_hat_nj 

seed(1)
nsim_temp = 50
pct_selected,coverage,int_length,beta_hat,Sigma_hat,pct_selected_nj,coverage_nj,int_length_nj,beta_hat_nj,Sigma_hat_nj = main(nsim=nsim_temp,p=p_temp)

print('Proportion of iterations with variables selected:')
print(pct_selected)

print('Signed difference in beta estimates')
print(np.mean(beta_hat - beta_hat_nj))
print('Absolute difference in beta estimates')
print(np.mean(abs(beta_hat - beta_hat_nj)))

print('Signed difference in se(beta_1):')
print(np.mean(Sigma_hat[:,0,0] - Sigma_hat_nj[:,0,0]))


