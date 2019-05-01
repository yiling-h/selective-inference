import numpy as np
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.bayesian.utils import glmnet_lasso, glmnet_lasso_cv1se, glmnet_lasso_cvmin, power_fdr
from selection.bayesian.generative_instance import generate_data, generate_data_new
from selection.bayesian.posterior_lasso import inference_lasso

def compare_inference(n= 65,
                      p= 1000,
                      sigma= 1.,
                      rho= 0.50,
                      randomizer_scale= 1.,
                      split_proportion= 0.60,
                      target="selected"):


    while True:
        X, y, beta, sigma, scalingX = generate_data_new(n=n, p=p, sigma=sigma, rho=rho, scale =True, center=True)
        n, p = X.shape

        true_set = np.asarray([u for u in range(p) if np.fabs(beta[u])> 1.])
        diff_set = np.fabs(np.subtract.outer(np.arange(p), np.asarray(true_set)))
        true_signals = np.asarray([x for x in range(p) if (min(diff_set[x, :]) <= 2).sum()>=1])

        if n>p:
            dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
            sigma_ = np.sqrt(dispersion)
        else:
            dispersion = None
            sigma_ = np.std(y)/np.sqrt(2.)
        print("sigma estimated and true ", sigma, sigma_)

        lam_theory = sigma_ * 0.80 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
        conv = lasso.gaussian(X,
                              y,
                              np.ones(X.shape[1]) * lam_theory,
                              randomizer_scale=randomizer_scale * sigma_)

        signs = conv.fit()
        nonzero = signs != 0
        nreport = 0.
        nactive = nonzero.sum()

        if nonzero.sum()>0:

            if target == "selected":
                beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = selected_targets(conv.loglike,
                                                  conv._W,
                                                  nonzero,
                                                  dispersion=dispersion)

            else:
                beta_target = beta[nonzero]
                (observed_target,
                 cov_target,
                 cov_target_score,
                 alternatives) = debiased_targets(conv.loglike,
                                                  conv._W,
                                                  nonzero,
                                                  penalty=conv.penalty,
                                                  dispersion=dispersion)

            active_screenset = np.asarray([r for r in range(p) if nonzero[r]])
            #print("beta ", beta, true_set, active_screenset, beta[true_set], beta[active_screenset])
            true_screen, false_screen = power_fdr(active_screenset, true_signals)

            power_screen = true_screen/float(true_set.sum())
            false_screen = false_screen/float(nactive)

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

            reportind = ~((lci < 0.) * (uci > 0.))
            reportset = np.asarray([active_screenset[a] for a in range(nactive) if reportind[a]==1])

            true_total, false_total = power_fdr(reportset, true_signals)
            power_total = true_total / float(true_set.sum())
            false_total = false_total / max(float(true_total+false_total), 1.)

            power_selective = true_total / max(float(true_screen), 1.)
            fdr_selective = false_total/max(float(true_total + false_total), 1.)

        else:
            nreport = 1.
            coverage, length, power_screen, false_screen, power_total, false_total, power_selective, fdr_selective =  [0., 0., 0., 0., 0., 0., 0., 0.]

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
        lam_theory_split = sigma_ * 0.80 * np.mean(
            np.fabs(np.dot(X_sel_scaled.T, np.random.standard_normal((subsample_size, 2000)))).max(0))
        glm_LASSO_split = glmnet_lasso(X_sel_scaled, y_sel, lam_theory_split / float(subsample_size))

        active_LASSO_split = (glm_LASSO_split != 0)
        nactive_split = active_LASSO_split.sum()

        if nactive_split>0:
            active_screenset_split = np.asarray([s for s in range(p) if active_LASSO_split[s]])
            true_screen_split, false_screen_split = power_fdr(active_screenset_split, true_signals)

            power_screen_split = true_screen_split / float(true_set.sum())
            false_screen_split = false_screen_split / float(nactive_split)

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

            reportind_split = ~((adjusted_intervals_split[0, :] < 0.) * (adjusted_intervals_split[1, :] > 0.))
            reportset_split = np.asarray([active_screenset_split[b] for b in range(nactive_split) if reportind_split[b] == 1])

            true_total_split, false_total_split = power_fdr(reportset_split, true_signals)
            power_total_split = true_total_split / float(true_set.sum())
            false_total_split = false_total_split / max(float(true_total_split + false_total_split),1.)

            power_selective_split = true_total_split / max(float(true_screen_split), 1.)
            fdr_selective_split = false_total_split / max(float(true_total_split + false_total_split), 1.)

        else:
            nreport_split = 1.
            coverage_split, length_split, power_screen_split, false_screen_split, \
            power_total_split, false_total_split, power_selective_split, fdr_selective_split = [0., 0., 0., 0., 0., 0., 0., 0.]

        return np.vstack((coverage, length, nactive, power_screen, false_screen,
                          power_total, false_total, power_selective, fdr_selective,
                          coverage_split, length_split, nactive_split, power_screen_split, false_screen_split,
                          power_total_split, false_total_split, power_selective_split, fdr_selective_split,
                          nreport, nreport_split))


def main(ndraw=10, split_proportion=0.70, randomizer_scale=1.):

    output = np.zeros(20)

    for n in range(ndraw):
        output += np.squeeze(compare_inference(n=65,
                                               p=300,
                                               sigma=1.,
                                               rho=0.50,
                                               randomizer_scale=randomizer_scale,
                                               split_proportion=split_proportion,
                                               target="debiased"))

        print("iteration completed ", n + 1)
    output /= (n+1.)
    print("inferential metrics so far ", output)
    #print("coverage  ", output[0] / (n + 1.-output[4]), output[2] / (n + 1.-output[5]))
    #print("lengths ", output[1] / (n + 1.-output[4]), output[3] / (n + 1.-output[5]))

main(ndraw=10)







