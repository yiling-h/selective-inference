import numpy as np
from scipy.stats import norm as ndist

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

from selection.tests.instance import gaussian_instance
from selection.tests.decorators import rpy_test_safe

from selection.randomized.screening import stepup, stepup_selection
from selection.randomized.randomization import randomization

def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)

@rpy_test_safe()
def test_BH_procedure():

    def BH_cutoff():
        Z = np.random.standard_normal(100)

        BH = stepup.BH(Z,
                       np.identity(100),
                       1.)

        cutoff = BH.stepup_Z / np.sqrt(2)
        return cutoff
    
    BH_cutoffs = BH_cutoff()

    for _ in range(50):
        Z = np.random.standard_normal(100)
        Z[:20] += 3

        np.testing.assert_allclose(sorted(BHfilter(2 * ndist.sf(np.fabs(Z)), q=0.2)),
                                   sorted(stepup_selection(Z, BH_cutoffs)[1]))

def test_independent_estimator(n=100, n1=80, q=0.2, signal=3, p=100):

    Z = np.random.standard_normal((n, p))
    Z[:, :10] += signal / np.sqrt(n)
    Z1 = Z[:n1]
    
    Zbar = np.mean(Z, 0)
    Zbar1 = np.mean(Z1, 0)
    perturb = Zbar1 - Zbar
    
    frac = n1 * 1. / n
    BH_select = stepup.BH(Zbar, np.identity(p) / n, np.sqrt((1 - frac) / (n * frac)), q=q)
    selected = BH_select.fit(perturb=perturb)
    
    observed_target = Zbar[selected]
    cov_target = np.identity(selected.sum()) / n
    cross_cov = -np.identity(p)[selected] / n

    observed_target1, cov_target1, cross_cov1, _ = BH_select.marginal_targets(selected)

    assert(np.linalg.norm(observed_target - observed_target1) / np.linalg.norm(observed_target) < 1.e-7)
    assert(np.linalg.norm(cov_target - cov_target1) / np.linalg.norm(cov_target) < 1.e-7)
    assert(np.linalg.norm(cross_cov - cross_cov1) / np.linalg.norm(cross_cov) < 1.e-7)

    (final_estimator, 
     _, 
     Z_scores, 
     pvalues, 
     intervals, 
     ind_unbiased_estimator) = BH_select.selective_MLE(observed_target, cov_target, cross_cov)

    Zbar2 = Z[n1:].mean(0)[selected]

    assert(np.linalg.norm(ind_unbiased_estimator - Zbar2) / np.linalg.norm(Zbar2) < 1.e-6)
    np.testing.assert_allclose(sorted(np.nonzero(selected)[0]), 
                               sorted(BHfilter(2 * ndist.sf(np.fabs(np.sqrt(n1) * Zbar1)))))


def test_BH(p=500,
            s=50,
            sigma=3.,
            rho=0.35,
            randomizer_scale=1.,
            use_MLE=True,
            marginal=False,
            level=0.9):

    while True:

        W = rho**(np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
        sqrtW = np.linalg.cholesky(W)
        Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
        beta = (2 * np.random.binomial(1, 0.5, size=(p,)) - 1) * np.linspace(3, 5, p) * sigma
        np.random.shuffle(beta)
        beta[s:] = 0
        np.random.shuffle(beta)
        print(beta, 'beta')

        true_mean = W.dot(beta)
        score = Z + true_mean

        q = 0.1
        BH_select = stepup.BH(score,
                              W * sigma**2,
                              randomizer_scale * sigma,
                              q=q)

        nonzero = BH_select.fit()

        if nonzero is not None:

            if marginal:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = BH_select.marginal_targets(nonzero)
            else:
                (observed_target, 
                 cov_target, 
                 crosscov_target_score, 
                 alternatives) = BH_select.full_targets(nonzero, dispersion=sigma**2)
               
            if marginal:
                beta_target = true_mean[nonzero]
            else:
                beta_target = beta[nonzero]

            if use_MLE:
                print('huh')
                estimate, info, _, pval, intervals, _ = BH_select.selective_MLE(observed_target,
                                                                                cov_target,
                                                                                crosscov_target_score,
                                                                                level=level)
                pivots = ndist.cdf((estimate - beta_target) / np.sqrt(np.diag(info)))
                pivots = 2 * np.minimum(pivots, 1 - pivots)
                # run summary
            else:
                pivots, pval, intervals = BH_select.summary(observed_target, 
                                                            cov_target, 
                                                            crosscov_target_score, 
                                                            alternatives,
                                                            compute_intervals=True,
                                                            level=level,
                                                            ndraw=20000,
                                                            burnin=2000,
                                                            parameter=beta_target)
            print(pval)
            print("beta_target and intervals", beta_target, intervals)
            coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
            print("coverage for selected target", coverage.sum()/float(nonzero.sum()))
            return pivots[beta_target == 0], pivots[beta_target != 0], coverage, intervals, pivots
        else:
            return [], [], [], [], []

def test_python_C(p=500,
                  s=20,
                  sigma=1.,
                  rho=0.35,
                  randomizer_scale=np.sqrt(0.50),
                  level=0.9,
                  marginal = True):

    W = rho ** (np.fabs(np.subtract.outer(np.arange(p), np.arange(p))))
    sqrtW = np.linalg.cholesky(W)
    Z = np.random.standard_normal(p).dot(sqrtW.T) * sigma
    beta = np.zeros(p)
    beta[:s] = (2 * np.random.binomial(1, 0.5, size=(s,)) - 1) * np.linspace(5, 5, s) * sigma
    np.random.shuffle(beta)

    true_mean = W.dot(beta)
    score = Z + true_mean

    q = 0.1
    omega = randomization.isotropic_gaussian((p,), randomizer_scale*sigma).sample()
    BH_select_py = stepup.BH(score,
                             W * sigma ** 2,
                             randomizer_scale * sigma,
                             q=q,
                             perturb = omega,
                             useC=False)

    nonzero = BH_select_py.fit()
    BH_select_C = stepup.BH(score,
                             W * sigma ** 2,
                             randomizer_scale * sigma,
                             q=q,
                             perturb= omega,
                             useC=True)
    nonzero_C = BH_select_C.fit()

    if nonzero.sum()>0:

        if marginal:
            beta_target = true_mean[nonzero]
            (observed_target,
             cov_target,
             crosscov_target_score,
             alternatives) = BH_select_py.marginal_targets(nonzero)
        else:
            beta_target = beta[nonzero]
            (observed_target,
             cov_target,
             crosscov_target_score,
             alternatives) = BH_select_py.full_targets(nonzero, dispersion=sigma ** 2)


        estimate_py, info_py, _, pval_py, intervals_py, _ = BH_select_py.selective_MLE(observed_target,
                                                                                       cov_target,
                                                                                       crosscov_target_score,
                                                                                       level=level)

        pivots_py = ndist.cdf((estimate_py - beta_target) / np.sqrt(np.diag(info_py)))
        pivots_py = 2 * np.minimum(pivots_py, 1 - pivots_py)

        estimate_C, info_C, _, pval_C, intervals_C, _ = BH_select_C.selective_MLE(observed_target,
                                                                                  cov_target,
                                                                                  crosscov_target_score,
                                                                                  level=level)
        
        pivots_C = ndist.cdf((estimate_C - beta_target) / np.sqrt(np.diag(info_C)))
        pivots_C = 2 * np.minimum(pivots_C, 1 - pivots_C)

        print("beta_target and intervals", beta_target, intervals_C, intervals_py)
        coverage_py = (beta_target > intervals_py[:, 0]) * (beta_target < intervals_py[:, 1])
        coverage_C = (beta_target > intervals_C[:, 0]) * (beta_target < intervals_C[:, 1])
        count = 0.
        if np.mean(coverage_py) != np.mean(coverage_C):
            print("coverage for target", np.mean(coverage_py), np.mean(coverage_C))
            count = 1.
        return pivots_py[beta_target == 0], pivots_py[beta_target != 0], coverage_py, intervals_py, pivots_py,\
               pivots_C[beta_target == 0], pivots_C[beta_target != 0], coverage_C, intervals_C, pivots_C, count
    else:
        return [], [], [], [], [], [], [], [], [], [], 0.

def compare_python_C(nsim =500, marginal = True):

    cover_py, length_int_py, cover_C, length_int_C = [], [], [], []
    count = 0
    for i in range(nsim):
        p0_py, pA_py, cover_py_, intervals_py, pivots_py, p0_C, pA_C, cover_C_, intervals_C, pivots_C, count_ = test_python_C(marginal = marginal)
        cover_py.extend(cover_py_)
        cover_C.extend(cover_C_)
        count += count_
        print(np.mean(cover_py), np.mean(cover_C), 'coverage so far')
        print('no of times where C and py solver differ', count)

compare_python_C(marginal= False)

def test_both():
    test_BH(marginal=True)
    test_BH(marginal=False)

def main(nsim=500, use_MLE=True, marginal=False):

    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    U = np.linspace(0, 1, 101)
    P0, PA, cover, length_int = [], [], [], []
    Ps = []
    for i in range(nsim):
        p0, pA, cover_, intervals, pivots = test_BH(use_MLE=use_MLE, marginal=marginal)
        Ps.extend(pivots)
        cover.extend(cover_)
        P0.extend(p0)
        PA.extend(pA)
        print(np.mean(cover),'coverage so far')

        period = 10
        if use_MLE:
            period = 50
        if i % period == 0 and i > 0:
            plt.clf()
            if len(P0) > 0:
                plt.plot(U, sm.distributions.ECDF(P0)(U), 'b', label='null')
            plt.plot(U, sm.distributions.ECDF(PA)(U), 'r', label='alt')
            plt.plot(U, sm.distributions.ECDF(Ps)(U), 'tab:orange', label='pivot')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.legend()
            plt.savefig('BH_pvals.pdf')

#main(nsim=500, use_MLE=True, marginal=True)


