from __future__ import division, print_function
import numpy as np

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats

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

def test_lasso_estimate(X, y, sigma, beta, randomizer_scale = 1.):

    while True:
        dispersion = None
        sigma_ = np.std(y)
        print("sigma ", sigma, sigma_)
        n, p = X.shape

        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:,1:].T, np.random.standard_normal((n, 2000)))).max(0))
        lasso_sol = lasso.gaussian(X,
                                   y,
                                   feature_weights= np.append(0.001,np.ones(p-1)* lam_theory),
                                   randomizer_scale= np.sqrt(n)* randomizer_scale * sigma_)

        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("solution", nonzero.sum(), nonzero[0])
        if nonzero.sum()>0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(lasso_sol.loglike,
                                              lasso_sol._W,
                                              nonzero,
                                              dispersion=dispersion)

            estimate, _, _, pval, intervals, _ = lasso_sol.selective_MLE(observed_target,
                                                                         cov_target,
                                                                         cov_target_score,
                                                                         alternatives)
            print("target and estimate", estimate[0], beta_target[0])
            return estimate[0]-beta_target[0], estimate[0]

def main(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=1, snr=0.20, randomizer_scale=0.5, nsim=100, B=10):

    carved_est = []
    bias = 0.
    mse = 0.
    for i in range(nsim):
        X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type,
                                                snr=snr)
        X -= X.mean(0)[None, :]
        X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
        y = y - y.mean()
        causal_estimate = 0.
        estimate = 0.
        mse = 0.
        for b in range(B):
            cc, est= test_lasso_estimate(X, y, sigma, beta, randomizer_scale=randomizer_scale)
            causal_estimate += cc
            estimate += est
        carved_est.append(causal_estimate/float(B))
        bias += (estimate/float(B)- beta[0])
        mse += ((estimate/float(B) - beta[0])**2)
        print("iteration completed ", i, '\n')
        print("bias so far ", bias/float(i+1), mse/float(i+1))

    #sns.distplot(np.asarray(carved_est))
    #plt.show()
    print("spread ", np.std(np.asarray(carved_est)))
    stats.probplot(np.asarray(carved_est), dist="norm", plot=pylab)
    pylab.show()

main(nsim=200, B=100)








