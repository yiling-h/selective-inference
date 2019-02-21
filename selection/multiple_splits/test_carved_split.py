from __future__ import division, print_function
import numpy as np, os

from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
import scipy.stats as stats
import pandas as pd
from selection.multiple_splits.utils import sim_xy, glmnet_lasso_cv1se, glmnet_lasso_cvmin, glmnet_lasso

def carved_estimate(X, y, sigma, randomizer_scale = 0.5, tuning="theory"):

    while True:
        dispersion = None
        sigma_ = np.std(y)
        print("sigma ", sigma, sigma_)
        n, p = X.shape

        if tuning== "theory":
            lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))
            lasso_sol = lasso.gaussian(X,
                                       y,
                                       feature_weights=np.append(0.001, np.ones(p - 1) * lam_theory),
                                       randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        elif tuning== "1se":
            glm_LASSO_1se, lam_1se = glmnet_lasso_cv1se(X, y)
            print("check ", lam_1se)
            lasso_sol = lasso.gaussian(X,
                                       y,
                                       feature_weights=np.append(0.001, np.ones(p - 1) * n * lam_1se),
                                       randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        else:
            glm_LASSO_min, lam_min = glmnet_lasso_cvmin(X, y)
            lasso_sol = lasso.gaussian(X,
                                       y,
                                       feature_weights=np.append(0.001, np.ones(p - 1) * n * lam_min),
                                       randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

        signs = lasso_sol.fit()
        nonzero = signs != 0
        print("solution", nonzero.sum(), nonzero[0])
        if nonzero.sum()>0:
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
            #print("target and estimate", estimate[0], beta_target[0])
            return estimate[0]

def create_output_file_carved(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=1, snr=0.20, randomizer_scale=0.5,
                              nsim=50, B_values=np.array([1, 10, 25, 100]), outpath="None"):

    df_mse = pd.DataFrame()

    for B in B_values:
        bias = 0.
        mse = 0.
        for i in range(nsim):
            print("snr", snr, n, p)
            X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type, snr=snr)
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
            y = y - y.mean()
            est = 0.
            for b in range(B):
                est += carved_estimate(X, y, sigma, randomizer_scale=randomizer_scale, tuning="1se")

            bias += (est/float(B) - beta[0])
            mse += (((est/float(B)) - beta[0]) ** 2)
            print("iteration completed ", i, '\n')
            print("bias and mse so far ", bias / float(i + 1), mse / float(i + 1))

        df_mse_iter = pd.DataFrame(data=np.vstack((bias/float(nsim), mse/float(nsim))).reshape((1, 2)),
                                   columns=['bias', 'MSE'])

        df_mse= df_mse.append(df_mse_iter, ignore_index=True)
        print("check ", df_mse)

    df_mse['n'] = n
    df_mse['p'] = p
    df_mse['s'] = s
    df_mse['rho'] = rho
    df_mse['beta-type'] = beta_type
    df_mse['snr'] = snr
    df_mse['B'] = pd.Series(B_values)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_mse_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_msebias_betatype" + str(beta_type) + "_rho_" + str(rho) + ".html")
    df_mse.to_html(outfile_mse_html)

create_output_file_carved(outpath='/Users/psnigdha/Research/Carving_causal_inference/Results/')

def split(X, y, sigma, split_fraction=0.67, tuning="theory"):
    while True:
        dispersion = None
        sigma_ = np.std(y)
        print("sigma ", sigma, sigma_)
        n, p = X.shape
        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X[:, 1:].T, np.random.standard_normal((n, 2000)))).max(0))

        subsample_size = int(split_fraction * n)
        sel_idx = np.zeros(n, np.bool)
        sel_idx[:subsample_size] = 1
        np.random.shuffle(sel_idx)
        inf_idx = ~sel_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        y_sel = y[sel_idx]
        X_sel = X[sel_idx, :]

        if tuning == "theory":
            glm_LASSO = glmnet_lasso(X_sel, y_sel, np.append(0.001, np.ones(p - 1) * lam_theory) / float(n))
            active_LASSO = (glm_LASSO != 0)

        elif tuning == "1se":
            glm_LASSO_1se, lam_1se = glmnet_lasso_cv1se(X_sel, y_sel)
            active_LASSO = (glm_LASSO_1se != 0)
            active_LASSO[0] = 1

        else:
            glm_LASSO_min, lam_min = glmnet_lasso_cvmin(X_sel, y_sel)
            active_LASSO = (glm_LASSO_min != 0)
            active_LASSO[0] = 1

        rel_LASSO = np.linalg.pinv(X_inf[:, active_LASSO]).dot(y_inf)
        return rel_LASSO[0]

def create_output_file_split(n=200, p=1000, nval=200, alpha= 2., rho=0.70, s=10, beta_type=1, snr=0.20, randomizer_scale=0.5,
                             nsim=50, B_values=np.array([1, 10, 25, 100]), outpath="None"):

    df_mse = pd.DataFrame()

    for B in B_values:
        bias = 0.
        mse = 0.
        for i in range(nsim):
            print("snr", snr, n, p)
            X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, alpha=alpha, rho=rho, s=s, beta_type=beta_type, snr=snr)
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
            y = y - y.mean()
            est = 0.
            for b in range(B):
                est += split(X, y, sigma, split_fraction = 0.67, tuning="min")

            bias += (est/float(B) - beta[0])
            mse += (((est/float(B)) - beta[0]) ** 2)
            print("iteration completed ", i, '\n')
            print("bias and mse so far ", bias / float(i + 1), mse / float(i + 1))

        df_mse_iter = pd.DataFrame(data=np.vstack((bias/float(nsim), mse/float(nsim))).reshape((1, 2)),
                                   columns=['bias', 'MSE'])

        df_mse= df_mse.append(df_mse_iter, ignore_index=True)
        print("check ", df_mse)

    df_mse['n'] = n
    df_mse['p'] = p
    df_mse['s'] = s
    df_mse['rho'] = rho
    df_mse['beta-type'] = beta_type
    df_mse['snr'] = snr
    df_mse['B'] = pd.Series(B_values)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_mse_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_msebias_betatype" + str(beta_type) + "_rho_" + str(rho) + ".html")
    df_mse.to_html(outfile_mse_html)

#create_output_file_split(outpath='/Users/psnigdha/Research/Carving_causal_inference/Results/')