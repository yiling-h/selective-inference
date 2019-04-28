import numpy as np, sys
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.bayesian.utils import inference_lasso, glmnet_lasso
from selection.bayesian.generative_instance import generate_data

def compare_inference(n= 65,
                      p= 1000,
                      sigma= 1.,
                      rho= 0.40,
                      randomizer_scale= 1.,
                      split_proportion= 0.60):


    while True:
        X, y, beta, sigma = generate_data(n=n, p=p, sigma=sigma, rho=rho, scale =True, center=True)
        n, p = X.shape

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)
        print("sigma estimated and true ", sigma, sigma_)

        lam_theory = sigma_ * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        conv = lasso.gaussian(X,
                              y,
                              np.ones(X.shape[1]) * lam_theory,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        nreport = 0.
        if nonzero.sum()>0:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)
            initial_par, _, _, _, _, _ = conv.selective_MLE(observed_target,
                                                            cov_target,
                                                            cov_target_score,
                                                            alternatives)

            posterior_inf = inference_lasso(observed_target,
                                            cov_target,
                                            cov_target_score,
                                            conv.observed_opt_state,
                                            conv.cond_mean,
                                            conv.cond_cov,
                                            conv.logdens_linear,
                                            conv.A_scaling,
                                            conv.b_scaling,
                                            initial_par)

            samples = posterior_inf.posterior_sampler(nsample=2000, nburnin=50)
            lci = np.percentile(samples, 5, axis=0)
            uci = np.percentile(samples, 95, axis=0)
            coverage = np.mean((lci < beta_target) * (uci > beta_target))
            length = np.mean(uci - lci)

        else:
            nreport = 1.
            coverage = 0.
            length = 0.

        nreport_split = 0.
        subsample_size = int(split_proportion * n)
        sel_idx = np.zeros(n, np.bool)
        sel_idx[:subsample_size] = 1
        np.random.shuffle(sel_idx)
        inf_idx = ~sel_idx
        y_inf = y[inf_idx]
        X_inf = X[inf_idx, :]
        y_sel = y[sel_idx]
        X_sel = X[sel_idx, :]
        X_sel_scaled = np.sqrt(((subsample_size - 1.) / float(subsample_size)) * n) * X_sel
        lam_theory_split = sigma_ * np.mean(
            np.fabs(np.dot(X_sel_scaled.T, np.random.standard_normal((subsample_size, 2000)))).max(0))
        glm_LASSO_split = glmnet_lasso(X_sel_scaled, y_sel, lam_theory_split / float(subsample_size))

        active_LASSO_split = (glm_LASSO_split != 0)
        nactive_split = active_LASSO_split.sum()

        if nactive_split>0:
            noise_variance = sigma_ ** 2.
            prior_variance = 100.
            active_LASSO_split = (glm_LASSO_split != 0)
            nactive_split = active_LASSO_split.sum()
            X_inf_split = X_inf[:, active_LASSO_split]
            projection_active_split = X_inf_split.dot(np.linalg.inv(X_inf_split.T.dot(X_inf_split)))
            true_val_split = projection_active_split.T.dot(X_inf.dot(beta))

            est_split = projection_active_split.T.dot(y_inf)
            M_split = np.linalg.inv(prior_variance * np.identity(nactive_split) + noise_variance * np.linalg.inv(
                X_inf_split.T.dot(X_inf_split)))
            post_mean_split = prior_variance * (M_split).dot(est_split)
            post_var_split = prior_variance * np.identity(nactive_split) - (prior_variance ** 2.) * (M_split)
            adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(post_var_split.diagonal())),
                                                  post_mean_split + 1.65 * (np.sqrt(post_var_split.diagonal()))])
            coverage_split = np.mean(
                (true_val_split > adjusted_intervals_split[0, :]) * (true_val_split < adjusted_intervals_split[1, :]))
            length_split = np.mean(adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :])

        else:
            nreport_split = 1.
            coverage_split = 0.
            length_split = 0.

        return np.vstack((coverage, length, coverage_split, length_split, nreport, nreport_split))


def main(ndraw=10, split_proportion=0.80, randomizer_scale=1.):

    output = np.zeros(6)

    for n in range(ndraw):
        output += np.squeeze(compare_inference(n=65,
                                               p=1000,
                                               sigma=1.,
                                               rho=0.40,
                                               randomizer_scale=randomizer_scale,
                                               split_proportion=split_proportion))

        print("iteration completed ", n + 1)

    print("coverage  ", output[0] / (n + 1.-output[4]), output[2] / (n + 1.-output[5]))
    print("lengths ", output[1] / (n + 1.-output[4]), output[3] / (n + 1.-output[5]))

main(ndraw=20)







