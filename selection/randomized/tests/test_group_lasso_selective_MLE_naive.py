import numpy as np

from selection.randomized.group_lasso import group_lasso
from selection.tests.instance import gaussian_group_instance, gaussian_instance
from selection.randomized.randomization import randomization

import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF
from scipy.stats import norm as ndist

def test_selected_targets(n=300,
                          p=100,
                          signal_fac=1/(2*np.log(100)),
                          sgroup=1,
                          #s=10,
                          #groups=np.arange(100),
                          groups=np.arange(50).repeat(2),
                          sigma=1.,
                          rho=0,
                          randomizer_scale=1.,
                          weight_frac=1.):

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
    conv = const(X, Y, groups, weights, randomizer_scale=randomizer_scale * sigma_)
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

    # naive inference
    if nonzero.sum() > 0:
        if n>p:
            dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (n - p)
        else:
            dispersion = sigma_ ** 2


        estimate_naive, observed_info_mean_naive, _, pval_naive, intervals_naive, _ = conv.naive_inference(dispersion=dispersion,
                                                                                                  level=0.9)

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        pivot_MLE_naive = ndist.cdf((estimate_naive - beta_target)/np.sqrt(np.diag(observed_info_mean_naive)))

        coverage_naive = (beta_target > intervals_naive[:, 0]) * (beta_target <
                                                             intervals_naive[:, 1])
        p0_naive = pval_naive[beta[nonzero] == 0]
        pa_naive = pval_naive[beta[nonzero] != 0]
    else:
        p0_naive = []
        pa_naive = []
        coverage_naive = []
        intervals_naive = []
        pivot_MLE_naive = []
        estimate_naive = np.zeros(p)
        observed_info_mean_naive = np.zeros((p,p))

    return p0,pa,coverage,intervals,pivot_MLE,estimate,observed_info_mean,p0_nj,pa_nj,coverage_nj,intervals_nj,pivot_MLE_nj,estimate_nj,observed_info_mean_nj,p0_naive,pa_naive,coverage_naive,intervals_naive,pivot_MLE_naive,estimate_naive,observed_info_mean_naive,


def main(nsim=100,snr=[.1*i for i in range(1,10)]):
    cover_snr, nj_cover_snr, naive_cover_snr = [], [], [] 
    avg_length_snr, nj_avg_length_snr, naive_avg_length_snr = [], [], []
    
    for s in snr:
        cover, pivot, avg_length = [], [], []
        nj_cover, nj_pivot, nj_avg_length = [], [], []
        naive_cover, naive_pivot, naive_avg_length = [], [], []
        for i in range(nsim):
            _,_, cover_, intervals, pivot_, _,_, _,_, nj_cover_, nj_intervals, nj_pivot_,_,_, _,_,naive_cover_, naive_intervals, naive_pivot_,_,_ = test_selected_targets(signal_fac = s)  
            if len(intervals)>0:
                avg_length_ = intervals[:, 1] - intervals[:, 0]
                nj_avg_length_ = nj_intervals[:,1] - nj_intervals[:,0]
                naive_avg_length_ = naive_intervals[:,1] - naive_intervals[:,0]
            else:
                avg_length_, nj_avg_length_, naive_avg_length_ = [], [], []

            cover.extend(cover_)
            pivot.extend(pivot_)
            avg_length.extend(avg_length_)
            print(np.mean(cover), np.mean(avg_length),
                  'coverage + length so far, selective MLE')
            
            nj_cover.extend(nj_cover_)
            nj_pivot.extend(nj_pivot_)
            nj_avg_length.extend(nj_avg_length_)
            print(np.mean(nj_cover), np.mean(nj_avg_length),
                  'coverage + length so far, selective MLE w/o Jacobian')
            
            naive_cover.extend(naive_cover_)
            naive_pivot.extend(naive_pivot_)
            naive_avg_length.extend(naive_avg_length_)
            print(np.mean(naive_cover), np.mean(naive_avg_length),
                  'coverage + length so far, naive')
        
        cover_snr.append(np.mean(cover))
        nj_cover_snr.append(np.mean(nj_cover))
        naive_cover_snr.append(np.mean(naive_cover))
        
        avg_length_snr.append(np.mean(avg_length))
        nj_avg_length_snr.append(np.mean(nj_avg_length))
        naive_avg_length_snr.append(np.mean(naive_avg_length))
            
        # plotting
        plt.clf()
        ecdf_MLE = ECDF(np.asarray(pivot))
        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_MLE(grid), c='blue', marker='^')
        plt.plot(grid, grid, 'k--')
        plt.show()
        
        plt.clf()
        ecdf_MLE = ECDF(np.asarray(nj_pivot))
        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_MLE(grid), c='green', marker='^')
        plt.plot(grid, grid, 'k--')
        plt.show()
        
        plt.clf()
        ecdf_MLE = ECDF(np.asarray(naive_pivot))
        grid = np.linspace(0, 1, 101)
        plt.plot(grid, ecdf_MLE(grid), c='red', marker='^')
        plt.plot(grid, grid, 'k--')
        plt.show()
    
    return cover_snr, nj_cover_snr, naive_cover_snr, avg_length_snr, nj_avg_length_snr, naive_avg_length_snr
    
    
cover, nj_cover, naive_cover, length, nj_length, naive_length = main(nsim=100,snr=[.1,.2,.3,.4])


