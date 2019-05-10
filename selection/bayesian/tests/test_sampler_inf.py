import numpy as np, os
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.bayesian.utils import glmnet_lasso, glmnet_lasso_cv1se, glmnet_lasso_cvmin, power_fdr
from selection.bayesian.generative_instance import generate_data, generate_data_instance
from selection.bayesian.posterior_lasso import inference_lasso

import pandas as pd

def sampler_inf(X, y, beta, detection_threshold, noise_threshold, randomizer_scale= 1., target="selected"):

    n, p = X.shape
    true_set = np.asarray([u for u in range(p) if np.fabs(beta[u]) >= detection_threshold])
    diff_set = np.fabs(np.subtract.outer(np.arange(p), np.asarray(true_set)))
    if true_set.shape[0] > 0:
        true_signals = np.asarray([x for x in range(p) if (min(diff_set[x, :]) <= 1).sum() >= 1])
    else:
        true_signals = np.asarray([])

    if n > p:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)

    lam_theory = sigma_ * 0.85 * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    conv = lasso.gaussian(X,
                          y,
                          lam_theory * np.ones(X.shape[1]),
                          randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()
    nonzero = signs != 0
    nactive = nonzero.sum()

    if nonzero.sum() > 0 and nonzero.sum() < 40:

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
        false_screenset = np.asarray([a for a in range(p) if (np.fabs(beta[a]) <= noise_threshold and nonzero[a])])

        true_screen = power_fdr(active_screenset, true_signals)
        false_screen = false_screenset.shape[0] / float(nactive)

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

        samples, count = posterior_inf.posterior_sampler(nsample=2000, nburnin=50, step=0.5, start=None,
                                                         Metropolis=False)

        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = (lci < beta_target) * (uci > beta_target)
        length = (uci - lci)

        reportind = ~((lci < 0.) * (uci > 0.))
        reportset = np.asarray([active_screenset[e] for e in range(nactive) if reportind[e] == 1])
        record_set = np.zeros(nactive)
        record_set[reportind] = reportset
        false_reportset = np.intersect1d(false_screenset, reportset)
        true_dtotal = power_fdr(reportset, true_signals)
        false_total = false_reportset.shape[0] / max(float(reportind.sum()), 1.)

        inference = np.vstack((coverage,
                               length,
                               active_screenset,
                               record_set,
                               reportind,
                               true_set.shape[0] * np.ones(nactive),
                               true_screen * np.ones(nactive),
                               false_screen * np.ones(nactive),
                               true_dtotal * np.ones(nactive),
                               false_total * np.ones(nactive)))

        return inference

    else:
        inference = np.zeros(10)
        inference[5] = true_set.shape[0]
        return inference

def split_inf(X, y, beta, detection_threshold, noise_threshold, split_proportion= 0.70, target="selected"):

    n, p = X.shape
    true_set = np.asarray([u for u in range(p) if np.fabs(beta[u]) >= detection_threshold])
    diff_set = np.fabs(np.subtract.outer(np.arange(p), np.asarray(true_set)))
    if true_set.shape[0] > 0:
        true_signals = np.asarray([x for x in range(p) if (min(diff_set[x, :]) <= 1).sum() >= 1])
    else:
        true_signals = np.asarray([])

    if n > p:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        sigma_ = np.std(y)

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
    lam_theory_split = sigma_ * 0.85 * np.mean(np.fabs(np.dot(X_sel_scaled.T, np.random.standard_normal((subsample_size, 2000)))).max(0))
    glm_LASSO_split = glmnet_lasso(X_sel_scaled, y_sel, lam_theory_split / float(subsample_size))

    active_LASSO_split = (glm_LASSO_split != 0)
    nactive_split = active_LASSO_split.sum()

    if nactive_split > 0 and nactive_split < 40:

        X_split = X[:, active_LASSO_split]
        active_screenset_split = np.asarray([s for s in range(p) if active_LASSO_split[s]])
        false_split_screenset = np.asarray([b for b in range(p) if (np.fabs(beta[b]) <= noise_threshold and active_LASSO_split[b])])
        true_screen_split = power_fdr(active_screenset_split, true_signals)
        false_screen_split = false_split_screenset.shape[0] / float(nactive_split)

        noise_variance = sigma_ ** 2.
        prior_variance = 100.
        active_LASSO_split = (glm_LASSO_split != 0)
        nactive_split = active_LASSO_split.sum()
        X_inf_split = X_inf[:, active_LASSO_split]
        projection_active_split = X_inf_split.dot(np.linalg.inv(X_inf_split.T.dot(X_inf_split)))
        true_val_split = np.linalg.pinv(X_split).dot(X.dot(beta))

        est_split = projection_active_split.T.dot(y_inf)
        M_split = np.linalg.inv(prior_variance * np.identity(nactive_split) + noise_variance * np.linalg.inv(
            X_inf_split.T.dot(X_inf_split)))
        post_mean_split = prior_variance * (M_split).dot(est_split)
        post_var_split = prior_variance * np.identity(nactive_split) - (prior_variance ** 2.) * (M_split)
        adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(np.diag(post_var_split))),
                                              post_mean_split + 1.65 * (np.sqrt(np.diag(post_var_split)))])

        coverage_split = (true_val_split > adjusted_intervals_split[0, :]) * (true_val_split < adjusted_intervals_split[1, :])
        length_split = adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :]

        reportind_split = ~((adjusted_intervals_split[0, :] < 0.) * (adjusted_intervals_split[1, :] > 0.))
        reportset_split = np.asarray(
            [active_screenset_split[f] for f in range(nactive_split) if reportind_split[f] == 1])
        record_set_split = np.zeros(nactive_split)
        record_set_split[reportind_split] = reportset_split
        false_reportset_split = np.intersect1d(false_split_screenset, reportset_split)

        true_dtotal_split = power_fdr(reportset_split, true_signals)
        false_total_split = false_reportset_split.shape[0] / max(float(reportind_split.sum()), 1.)


        inference = np.vstack((coverage_split,
                               length_split,
                               active_screenset_split,
                               record_set_split,
                               reportind_split,
                               true_set.shape[0] * np.ones(nactive_split),
                               true_screen_split * np.ones(nactive_split),
                               false_screen_split * np.ones(nactive_split),
                               true_dtotal_split * np.ones(nactive_split),
                               false_total_split * np.ones(nactive_split)))

        return inference

    else:
        inference = np.zeros(10)
        inference[5] = true_set.shape[0]
        return inference

def create_output(n=65,
                  p=350,
                  sigma=1.,
                  rho=0.30,
                  V_values=np.array([1., 1.5, 2., 2.5, 3, 4.5, 6.]),
                  randomizer_scale=1.,
                  split_proportion=0.70,
                  target="selected",
                  nsim = 100,
                  outpath = None):

    df_master = pd.DataFrame()

    for V in V_values:
        for ndraw in range(nsim):

            X, y, beta, sigma, _ = generate_data(n=n, p=p, sigma=sigma, rho=rho, V=V, scale=True, center=True)
            _, p = X.shape
            try:
                output_selective = (sampler_inf(X,
                                                y,
                                                beta,
                                                detection_threshold=np.sqrt(0.5 * 2 * np.log(p)),
                                                noise_threshold=0.1,
                                                randomizer_scale=randomizer_scale,
                                                target=target)).T

                df_sel = pd.DataFrame(data=output_selective, columns=['coverage', 'length', 'activeset',
                                                                      'reportset', 'reportind', 'ntrue',
                                                                      'true_screen', 'false_screen',
                                                                      'true_total', 'false_total'])

                df_sel = df_sel.assign(simulation = ndraw,
                                       method = "selective",
                                       snr = V)

                output_split = (split_inf(X,
                                          y,
                                          beta,
                                          detection_threshold=np.sqrt(0.5 * 2 * np.log(p)),
                                          noise_threshold=0.1,
                                          split_proportion=split_proportion,
                                          target=target)).T

                df_split = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'activeset',
                                                                    'reportset', 'reportind', 'ntrue',
                                                                    'true_screen', 'false_screen',
                                                                    'true_total', 'false_total'])

                df_split = df_split.assign(simulation = ndraw,
                                           method = "split",
                                           snr = V)

                df_master = df_master.append(df_sel, ignore_index=True)
                df_master = df_master.append(df_split, ignore_index=True)


            except ValueError:
                pass

            print("iteration completed ", ndraw + 1)



    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_" + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_" + target + "_rho_" + str(rho) + ".html")
    df_master.to_csv(outfile_inf_csv, index=False)
    df_master.to_html(outfile_inf_html)

create_output(nsim=50, outpath = None)




