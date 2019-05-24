import numpy as np, os
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.bayesian.utils import glmnet_lasso, glmnet_lasso_cv1se, glmnet_lasso_cvmin, power_fdr, discoveries_count
from selection.bayesian.posterior_lasso import inference_lasso
from scipy.stats import norm as ndist
from selection.bayesian.generative_instance import generate_signals
import pandas as pd

def sampler_inf(X,
                y,
                beta,
                true_clusters,
                false_clusters,
                clusters,
                detection_threshold,
                randomizer_scale,
                target,
                lam_frac = 1.1):

    n, p = X.shape
    if n > p:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)

    lam_theory = sigma_ * lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    conv = lasso.gaussian(X,
                          y,
                          lam_theory * np.ones(X.shape[1]),
                          randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()
    nonzero = signs != 0
    nactive = nonzero.sum()

    if nonzero.sum() > 0 and nonzero.sum() < 30:
        try:
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
            true_screen, false_screen, tot_screen = discoveries_count(active_screenset, true_clusters, false_clusters,
                                                                      clusters)

            power_screen = true_screen / max(float(len(true_clusters)), 1.)
            false_screen = false_screen / float(tot_screen)

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

            samples, count = posterior_inf.posterior_sampler(nsample=2200, nburnin=200, step=0.20, start=None,
                                                             Metropolis=False)

            lci = np.percentile(samples, 5, axis=0)
            uci = np.percentile(samples, 95, axis=0)
            coverage = (lci < beta_target) * (uci > beta_target)
            length = uci - lci

            reportind = np.zeros(nactive, np.bool)
            for s in range(nactive):
                if (np.mean(samples[:, s] > detection_threshold) > 0.50 or np.mean(
                        samples[:, s] < -detection_threshold) > 0.50):
                    reportind[s] = 1

            reportset = np.asarray([active_screenset[e] for e in range(nactive) if reportind[e] == 1])

            true_dtotal, false_dtotal, dtotal = discoveries_count(reportset, true_clusters, false_clusters, clusters)
            power_total = true_dtotal / max(float(len(true_clusters)), 1.)
            false_total = false_dtotal / max(float(dtotal), 1.)

            power_selective = true_dtotal / max(float(true_screen), 1.)
            ndiscoveries = reportind.sum()

            return np.vstack((coverage,
                              length,
                              nactive * np.ones(nactive),
                              true_screen * np.ones(nactive),
                              power_screen * np.ones(nactive),
                              false_screen * np.ones(nactive),
                              power_total * np.ones(nactive),
                              false_total * np.ones(nactive),
                              power_selective * np.ones(nactive),
                              ndiscoveries * np.ones(nactive),
                              true_dtotal * np.ones(nactive)))

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                return np.zeros(11)

    else:
        return np.zeros(11)


def split_inf(X,
              y,
              beta,
              true_clusters,
              false_clusters,
              clusters,
              detection_threshold,
              split_proportion,
              target,
              lam_frac = 1.1):

    n, p = X.shape
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
    lam_theory_split = sigma_ * lam_frac * np.mean(np.fabs(np.dot(X_sel_scaled.T, np.random.standard_normal((subsample_size, 2000)))).max(0))
    glm_LASSO_split = glmnet_lasso(X_sel_scaled, y_sel, lam_theory_split / float(subsample_size))
    active_LASSO_split = (glm_LASSO_split != 0)
    nactive_split = active_LASSO_split.sum()

    if nactive_split > 0 and nactive_split < 30:

        X_split = X[:, active_LASSO_split]
        active_screenset_split = np.asarray([s for s in range(p) if active_LASSO_split[s]])

        true_screen_split, false_screen_split, tot_screen_split = discoveries_count(active_screenset_split,
                                                                                    true_clusters,
                                                                                    false_clusters, clusters)

        power_screen_split = true_screen_split / max(float(len(true_clusters)), 1.)
        false_screen_split = false_screen_split / float(tot_screen_split)

        noise_variance = sigma_ ** 2.
        prior_variance = 100.
        active_LASSO_split = (glm_LASSO_split != 0)
        nactive_split = active_LASSO_split.sum()
        X_inf_split = X_inf[:, active_LASSO_split]
        try:
            projection_active_split = X_inf_split.dot(np.linalg.inv(X_inf_split.T.dot(X_inf_split)))
            true_val_split = np.linalg.pinv(X_split).dot(X.dot(beta))

            est_split = projection_active_split.T.dot(y_inf)
            M_split = np.linalg.inv(prior_variance * np.identity(nactive_split) + noise_variance * np.linalg.inv(
                X_inf_split.T.dot(X_inf_split)))
            post_mean_split = prior_variance * (M_split).dot(est_split)
            post_var_split = prior_variance * np.identity(nactive_split) - (prior_variance ** 2.) * (M_split)
            adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(np.diag(post_var_split))),
                                                  post_mean_split + 1.65 * (np.sqrt(np.diag(post_var_split)))])

            coverage_split = (true_val_split > adjusted_intervals_split[0, :]) * (
                        true_val_split < adjusted_intervals_split[1, :])
            length_split = adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :]

            reportind_split = np.zeros(nactive_split, np.bool)
            posterior_mass_pos = ndist.cdf((detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
            posterior_mass_neg = ndist.cdf(
                (-detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
            for u in range(nactive_split):
                if 1. - posterior_mass_pos[u] > 0.50 or posterior_mass_neg[u] > 0.50:
                    reportind_split[u] = 1

            reportset_split = np.asarray(
                [active_screenset_split[f] for f in range(nactive_split) if reportind_split[f] == 1])

            true_dtotal_split, false_dtotal_split, dtotal_split = discoveries_count(reportset_split, true_clusters,
                                                                                    false_clusters, clusters)
            power_total_split = true_dtotal_split / max(float(len(true_clusters)), 1.)
            false_total_split = false_dtotal_split / max(float(dtotal_split), 1.)

            power_selective_split = true_dtotal_split / max(float(true_screen_split), 1.)
            ndiscoveries_split = reportind_split.sum()

            return np.vstack((coverage_split,
                              length_split,
                              nactive_split * np.ones(nactive_split),
                              true_screen_split * np.ones(nactive_split),
                              power_screen_split * np.ones(nactive_split),
                              false_screen_split * np.ones(nactive_split),
                              power_total_split * np.ones(nactive_split),
                              false_total_split * np.ones(nactive_split),
                              power_selective_split * np.ones(nactive_split),
                              ndiscoveries_split * np.ones(nactive_split),
                              true_dtotal_split * np.ones(nactive_split)))

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                return np.zeros(11)

    else:
        return np.zeros(11)

def create_output(V_values=np.array([5.5, 6.5]),
                  randomizer_scale=1.,
                  split_proportion=0.70,
                  target="selected",
                  nsim = 100,
                  inpath = "/Users/psnigdha/Research/RadioiBAG/Data/",
                  outpath = None):

    df_master = pd.DataFrame()

    for V in V_values:
        for ndraw in range(nsim):

            X, y, beta, sigma, true_clusters, false_clusters, clusters, detection_threshold = generate_signals(inpath, V=V)

            try:
                output_selective = (sampler_inf(X = X,
                                                y = y,
                                                beta = beta,
                                                true_clusters = true_clusters,
                                                false_clusters = false_clusters,
                                                clusters = clusters,
                                                detection_threshold = detection_threshold,
                                                randomizer_scale = randomizer_scale,
                                                target=target)).T

                df_sel = pd.DataFrame(data=output_selective, columns=['coverage', 'length', 'nactive',
                                                                    'true_screen', 'power_screen', 'false_screen',
                                                                    'power_total', 'false_total', 'power_selective',
                                                                    'ndiscoveries', 'true_dtotal'])

                df_sel = df_sel.assign(simulation = ndraw,
                                       method = "selective",
                                       snr = V)

                output_split = (split_inf(X = X,
                                          y = y,
                                          beta = beta,
                                          true_clusters = true_clusters,
                                          false_clusters = false_clusters,
                                          clusters = clusters,
                                          detection_threshold = detection_threshold,
                                          split_proportion = split_proportion,
                                          target=target)).T

                df_split = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                    'true_screen', 'power_screen', 'false_screen',
                                                                    'power_total', 'false_total', 'power_selective',
                                                                    'ndiscoveries', 'true_dtotal'])

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

    outfile_inf_csv = os.path.join(outpath, "realX" + "_inference_35_high_90_" + target + ".csv")
    outfile_inf_html = os.path.join(outpath, "realX" + "_inference_35_high_90_" + target + ".html")
    df_master.to_csv(outfile_inf_csv, index=False)
    df_master.to_html(outfile_inf_html)

create_output(nsim=50, outpath = "/Users/psnigdha/Research/RadioiBAG/Results/")

def create_split_output(V_values=np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
                        target="selected",
                        nsim = 100,
                        inpath = "/Users/psnigdha/Research/RadioiBAG/Data/",
                        outpath = None):

    df_master = pd.DataFrame()

    for V in V_values:
        for ndraw in range(nsim):

            X, y, beta, sigma, true_clusters, false_clusters, clusters, detection_threshold = generate_signals(inpath, V=V)

            try:

                output_split = (split_inf(X = X,
                                          y = y,
                                          beta = beta,
                                          true_clusters = true_clusters,
                                          false_clusters = false_clusters,
                                          clusters = clusters,
                                          detection_threshold = detection_threshold,
                                          split_proportion = 0.50,
                                          target=target)).T

                df_split_1 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                      'true_screen', 'power_screen', 'false_screen',
                                                                      'power_total', 'false_total', 'power_selective',
                                                                      'ndiscoveries', 'true_dtotal'])

                df_split_1 = df_split_1.assign(simulation = ndraw,
                                               method = "split (50%)",
                                               snr = V)

                output_split = (split_inf(X=X,
                                          y=y,
                                          beta=beta,
                                          true_clusters=true_clusters,
                                          false_clusters=false_clusters,
                                          clusters=clusters,
                                          detection_threshold=detection_threshold,
                                          split_proportion=0.60,
                                          target=target)).T

                df_split_2 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                      'true_screen', 'power_screen', 'false_screen',
                                                                      'power_total', 'false_total', 'power_selective',
                                                                      'ndiscoveries', 'true_dtotal'])

                df_split_2 = df_split_2.assign(simulation=ndraw,
                                               method="split (60%)",
                                               snr=V)

                output_split = (split_inf(X=X,
                                          y=y,
                                          beta=beta,
                                          true_clusters=true_clusters,
                                          false_clusters=false_clusters,
                                          clusters=clusters,
                                          detection_threshold=detection_threshold,
                                          split_proportion=0.80,
                                          target=target)).T

                df_split_3 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                      'true_screen', 'power_screen', 'false_screen',
                                                                      'power_total', 'false_total', 'power_selective',
                                                                      'ndiscoveries', 'true_dtotal'])

                df_split_3 = df_split_3.assign(simulation=ndraw,
                                               method="split (80%)",
                                               snr=V)

                output_split = (split_inf(X=X,
                                          y=y,
                                          beta=beta,
                                          true_clusters=true_clusters,
                                          false_clusters=false_clusters,
                                          clusters=clusters,
                                          detection_threshold=detection_threshold,
                                          split_proportion=0.90,
                                          target=target)).T

                df_split_4 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                      'true_screen', 'power_screen', 'false_screen',
                                                                      'power_total', 'false_total', 'power_selective',
                                                                      'ndiscoveries', 'true_dtotal'])

                df_split_4 = df_split_4.assign(simulation=ndraw,
                                               method="split (90%)",
                                               snr=V)

                df_master = df_master.append(df_split_1, ignore_index=True)
                df_master = df_master.append(df_split_2, ignore_index=True)
                df_master = df_master.append(df_split_3, ignore_index=True)
                df_master = df_master.append(df_split_4, ignore_index=True)

            except ValueError:
                pass

            print("iteration completed ", ndraw + 1)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "realX" + "_inference_35_split_90_" + target + ".csv")
    outfile_inf_html = os.path.join(outpath, "realX" + "_inference_35_split_90_" + target + ".html")
    df_master.to_csv(outfile_inf_csv, index=False)
    df_master.to_html(outfile_inf_html)

#create_split_output(nsim=50, outpath = "/Users/psnigdha/Research/RadioiBAG/Results/")
