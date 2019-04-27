import numpy as np, sys
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance
from selection.bayesian.utils import inference_lasso, glmnet_lasso


def test_approx_pivot(n= 500,
                      p= 100,
                      signal_fac= 1.,
                      s= 5,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.,
                      split_proportion=0.70):

    inst = gaussian_instance
    signal = np.sqrt(signal_fac * 2. * np.log(p))

    while True:
        X, y, beta = inst(n=n,
                          p=p,
                          signal=signal,
                          s=s,
                          equicorrelated=False,
                          rho=rho,
                          sigma=sigma,
                          random_signs=True)[:3]

        n, p = X.shape

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)

        print("sigma estimated and true ", sigma, sigma_)

        #W = np.ones(X.shape[1]) * np.sqrt(2 * np.log(p)) * sigma_
        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

        conv = lasso.gaussian(X,
                              y,
                              np.ones(X.shape[1]) * lam_theory,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        (observed_target,
         cov_target,
         cov_target_score,
         alternatives) = selected_targets(conv.loglike,
                                          conv._W,
                                          nonzero,
                                          dispersion=dispersion)


        posterior_inf = inference_lasso(observed_target,
                                        cov_target,
                                        cov_target_score,
                                        conv.observed_opt_state,
                                        conv.cond_mean,
                                        conv.cond_cov,
                                        conv.logdens_linear,
                                        conv.A_scaling,
                                        conv.b_scaling)

        samples = posterior_inf.posterior_sampler(np.zeros(nonzero.sum()))
        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = np.mean((lci < beta_target) * (uci > beta_target))
        length = np.mean(uci - lci)
        #print("sample quantiles ", np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0))
        #print("coverage, lengths ", coverage_split, length_split)

        subsample_size = int(split_proportion * n)
        inference_size = n - subsample_size
        sel_idx = np.zeros(n, np.bool)
        sel_idx[:subsample_size] = 1
        np.random.shuffle(sel_idx)
        inf_idx = ~sel_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        y_sel = y[sel_idx]
        X_sel = X[sel_idx, :]
        #lam_theory_split = sigma_ * np.mean(np.fabs(np.dot(X_sel.T, np.random.standard_normal((subsample_size, 2000)))).max(0))
        #glm_LASSO_split = glmnet_lasso(np.sqrt(subsample_size) * X_sel, y_sel, lam_theory /float(subsample_size))
        glm_LASSO_split = nonzero
        noise_variance = sigma_**2.
        prior_variance = 100.
        active_LASSO_split = (glm_LASSO_split != 0)
        print("check ", active_LASSO_split.sum())
        projection_active_split = X_inf[:, active_LASSO_split].dot(np.linalg.inv(X_inf[:, active_LASSO_split].T.dot(X_inf[:, active_LASSO_split])))
        true_val_split = projection_active_split.T.dot(X[inf_idx, :].dot(beta))

        M_1_split = prior_variance * (X_inf.dot(X_inf.T)) + noise_variance * np.identity(int(inference_size))
        M_2_split = prior_variance * ((X_inf.dot(X_inf.T)).dot(projection_active_split))
        M_3_split = prior_variance * (
            projection_active_split.T.dot(X_inf.dot(X_inf.T)).dot(projection_active_split))
        post_mean_split = M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(y_inf)
        post_var_split = M_3_split - M_2_split.T.dot(np.linalg.inv(M_1_split)).dot(M_2_split)
        adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(post_var_split.diagonal())),
                                              post_mean_split + 1.65 * (np.sqrt(post_var_split.diagonal()))])
        coverage_split = np.mean((true_val_split > adjusted_intervals_split[0, :]) * (true_val_split < adjusted_intervals_split[1, :]))
        length_split = np.mean(adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :])

        return coverage, length, coverage_split, length_split


def main(ndraw=10, split_proportion=0.60, randomizer_scale=1.):

    coverage_ = 0.
    length_ = 0.
    coverage_split_ = 0.
    length_split_ = 0.

    for n in range(ndraw):
        cov, len, cov_split, len_split = test_approx_pivot(n=65,
                                                           p=1000,
                                                           signal_fac=0.6,
                                                           s=10,
                                                           sigma=1.,
                                                           rho=0.40,
                                                           randomizer_scale=randomizer_scale,
                                                           split_proportion=split_proportion)

        coverage_ += cov
        length_ += len
        coverage_split_ += cov_split
        length_split_ += len_split

        print("coverage so far ", coverage_ / (n + 1.), coverage_split_ / (n + 1.))
        print("lengths so far ", length_ / (n + 1.), length_split_/ (n + 1.))
        print("iteration completed ", n + 1)

main(ndraw=20)


