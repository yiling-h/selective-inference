import numpy as np, os
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.bayesian.utils import glmnet_lasso, glmnet_lasso_cv1se, glmnet_lasso_cvmin, power_fdr, discoveries_count
from selection.bayesian.posterior_lasso import inference_lasso, inference_lasso_hierarchical
from scipy.stats import norm as ndist
from selection.bayesian.generative_instance import generate_signals
import pandas as pd
import timeit

def naive_inf(X,
              y,
              beta,
              true_signals,
              false_signals,
              detection_threshold,
              lam_theory,
              sigma_,
              model_size = 50):

    n, p = X.shape
    y_inf = y
    X_inf = X
    y_sel = y
    X_sel = X
    try:
        lasso = lasso_full.gaussian(X_sel, y_sel, lam_theory)
        LASSO_split = lasso.fit()
        active_LASSO_split = (LASSO_split != 0)
        nactive_split = active_LASSO_split.sum()

        if nactive_split > 0 and nactive_split < model_size:

            X_split = X[:, active_LASSO_split]
            active_screenset_split = np.asarray([s for s in range(p) if active_LASSO_split[s]])

            true_screen_split, false_screen_split, tot_screen_split = discoveries_count(active_screenset_split,
                                                                                        true_signals,
                                                                                        false_signals,
                                                                                        X)

            power_screen_split = true_screen_split / max(float(true_signals.shape[0]), 1.)
            false_screen_split = false_screen_split / float(tot_screen_split)

            noise_variance = sigma_ ** 2.
            prior_variance = 1000.
            active_LASSO_split = (LASSO_split != 0)
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
                print("check ", true_val_split, adjusted_intervals_split, coverage_split, sigma_)

                length_split = adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :]

                reportind_split = np.zeros(nactive_split, np.bool)
                posterior_mass_pos = ndist.cdf(
                    (detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
                posterior_mass_neg = ndist.cdf(
                    (-detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
                for u in range(nactive_split):
                    if 1. - posterior_mass_pos[u] > 0.50 or posterior_mass_neg[u] > 0.50:
                        reportind_split[u] = 1

                reportset_split = np.asarray(
                    [active_screenset_split[f] for f in range(nactive_split) if reportind_split[f] == 1])

                true_dtotal_split, false_dtotal_split, dtotal_split = discoveries_count(reportset_split,
                                                                                        true_signals,
                                                                                        false_signals,
                                                                                        X)

                power_total_split = true_dtotal_split / max(float(true_signals.shape[0]), 1.)
                false_total_split = false_dtotal_split / max(float(dtotal_split), 1.)

                power_selective_split = true_dtotal_split / max(float(true_screen_split), 1.)
                ndiscoveries_split = reportind_split.sum()

                return np.vstack((np.mean(coverage_split),
                                  np.mean(length_split),
                                  nactive_split ,
                                  true_screen_split ,
                                  power_screen_split ,
                                  false_screen_split ,
                                  power_total_split ,
                                  false_total_split ,
                                  power_selective_split ,
                                  ndiscoveries_split,
                                  true_dtotal_split,
                                  true_signals.shape[0]))

            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    return np.zeros((12, 1))

        else:
            return np.zeros((12,1))

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            return np.zeros((12, 1))


def split_inf(X,
              y,
              beta,
              true_signals,
              false_signals,
              detection_threshold,
              split_proportion,
              lam_theory,
              sigma_,
              model_size = 20):

    n, p = X.shape
    subsample_size = int(split_proportion * n)
    sel_idx = np.zeros(n, np.bool)
    sel_idx[:subsample_size] = 1
    np.random.shuffle(sel_idx)
    inf_idx = ~sel_idx
    y_inf = y[inf_idx]
    X_inf = X[inf_idx, :]
    y_sel = y[sel_idx]
    X_sel = X[sel_idx, :]
    try:
        lasso = lasso_full.gaussian(X_sel, y_sel, lam_theory)
        LASSO_split = lasso.fit()
        active_LASSO_split = (LASSO_split != 0)
        nactive_split = active_LASSO_split.sum()

        if nactive_split > 0 and nactive_split < model_size:

            X_split = X[:, active_LASSO_split]
            active_screenset_split = np.asarray([s for s in range(p) if active_LASSO_split[s]])

            true_screen_split, false_screen_split, tot_screen_split = discoveries_count(active_screenset_split,
                                                                                        true_signals,
                                                                                        false_signals,
                                                                                        X)

            if true_signals.shape[0] > 0:
                power_screen_split = true_screen_split / max(float(true_signals.shape[0]), 1.)
            else:
                power_screen_split = 1

            false_screen_split = false_screen_split / float(tot_screen_split)

            noise_variance = sigma_ ** 2.
            prior_variance = 100.
            active_LASSO_split = (LASSO_split != 0)
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
                posterior_mass_pos = ndist.cdf(
                    (detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
                posterior_mass_neg = ndist.cdf(
                    (-detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
                for u in range(nactive_split):
                    if 1. - posterior_mass_pos[u] > 0.50 or posterior_mass_neg[u] > 0.50:
                        reportind_split[u] = 1

                reportset_split = np.asarray(
                    [active_screenset_split[f] for f in range(nactive_split) if reportind_split[f] == 1])

                true_dtotal_split, false_dtotal_split, dtotal_split = discoveries_count(reportset_split,
                                                                                        true_signals,
                                                                                        false_signals,
                                                                                        X)

                if true_signals.shape[0]>0:
                    power_total_split = true_dtotal_split / max(float(true_signals.shape[0]), 1.)
                else:
                    power_total_split = 1.
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
                                  true_dtotal_split * np.ones(nactive_split),
                                  true_signals.shape[0]*np.ones(nactive_split)))

            except np.linalg.LinAlgError as e:
                if 'Singular matrix' in str(e):
                    return np.zeros((12, 1))

        else:
            return np.zeros((12,1))

    except np.linalg.LinAlgError as e:
        if 'Singular matrix' in str(e):
            return np.zeros((12, 1))


def sampler_inf(X,
                y,
                beta,
                true_signals,
                false_signals,
                detection_threshold,
                randomizer_scale,
                target,
                lam_theory,
                sigma_,
                dispersion=None,
                step_size=1.,
                model_size=65):
    n, p = X.shape

    conv = lasso.gaussian(X,
                          y,
                          lam_theory * np.ones(X.shape[1]),
                          randomizer_scale=randomizer_scale * sigma_)

    signs = conv.fit()
    nonzero = signs != 0
    nactive = nonzero.sum()

    if nonzero.sum() > 0 and nonzero.sum() < model_size:
        try:
            beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(conv.loglike,
                                              conv._W,
                                              nonzero,
                                              dispersion=dispersion)

            active_screenset = np.asarray([r for r in range(p) if nonzero[r]])
            true_screen, false_screen, tot_screen = discoveries_count(active_screenset, true_signals, false_signals,
                                                                      X)
            if true_signals.shape[0] > 0:
                power_screen = true_screen / max(float(true_signals.shape[0]), 1.)
            else:
                power_screen = 1.

            false_screen = false_screen / float(tot_screen)

            initial_par, observed_info_mean, _, _, _, _ = conv.selective_MLE(observed_target,
                                                            cov_target,
                                                            cov_target_score,
                                                            alternatives)

            posterior_inf = inference_lasso_hierarchical(observed_target,
                                                         cov_target,
                                                         cov_target_score,
                                                         conv.observed_opt_state,
                                                         conv.cond_mean,
                                                         conv.cond_cov,
                                                         conv.logdens_linear,
                                                         conv.A_scaling,
                                                         conv.b_scaling,
                                                         initial_par)

            tic= timeit.default_timer()

            samples, count = posterior_inf.posterior_sampler(nsample=2200, nburnin=200, step=step_size, start=None,
                                                             Metropolis=False)

            toc = timeit.default_timer()
            time_ = toc - tic

            samples = samples[:, :nactive]

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

            true_dtotal, false_dtotal, dtotal = discoveries_count(reportset, true_signals, false_signals, X)
            if true_signals.shape[0]>0:
                power_total = true_dtotal / max(float(true_signals.shape[0]), 1.)
            else:
                power_total = 1.
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
                              true_dtotal * np.ones(nactive),
                              true_signals.shape[0] * np.ones(nactive),
                              time_* np.ones(nactive)))

        except np.linalg.LinAlgError as e:
            if 'Singular matrix' in str(e):
                return np.zeros(13)

    else:
        return np.zeros(13)


def create_output(V_values,
                  nsim,
                  inpath,
                  null_prob_values,
                  lam_frac_values,
                  step_size_values,
                  randomizer_scale=1.,
                  target="selected",
                  outpath = None):

    df_master = pd.DataFrame()

    for j in range(V_values.shape[0]):
        for ndraw in range(nsim):
            V = V_values[j]
            null_prob = null_prob_values[j]
            X, y, beta, sigma, true_signals, false_signals, detection_threshold = generate_signals(inpath,
                                                                                                   V=V,
                                                                                                   null_prob=null_prob)

            n, p = X.shape

            if n > 2 * p:
                dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
                sigma_ = np.sqrt(dispersion)
            else:
                sigma_ = np.std(y)

            lam_theory = sigma_ * lam_frac_values[j] * np.mean(
                np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

            try:
                output_selective = (sampler_inf(X=X,
                                                y=y,
                                                beta=beta,
                                                true_signals=true_signals,
                                                false_signals=false_signals,
                                                detection_threshold=detection_threshold,
                                                randomizer_scale=randomizer_scale,
                                                target=target,
                                                lam_theory=lam_theory,
                                                sigma_=sigma_,
                                                step_size=step_size_values[j])).T

                df_sel = pd.DataFrame(data=output_selective, columns=['coverage', 'length', 'nactive',
                                                                      'true_screen', 'power_screen', 'false_screen',
                                                                      'power_total', 'false_total', 'power_selective',
                                                                      'ndiscoveries', 'true_dtotal', 'true_total', 'run_time'])

                df_sel = df_sel.assign(simulation=ndraw,
                                       method="selective",
                                       snr=V,
                                       null_probab=null_prob)

                df_master = df_master.append(df_sel, ignore_index=True)

            except ValueError:
                pass

            print("iteration completed ", ndraw + 1)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "realX_low_PF" + "_inference_35_90_" + target + ".csv")
    outfile_inf_html = os.path.join(outpath, "realX_low_PF" + "_inference_35_90_" + target + ".html")
    df_master.to_csv(outfile_inf_csv, index=False)
    df_master.to_html(outfile_inf_html)

def create_split_output(V_values,
                        nsim,
                        inpath,
                        null_prob_values,
                        lam_frac_values,
                        target="selected",
                        outpath = None):

    df_master = pd.DataFrame()

    for j in range(V_values.shape[0]):
        for ndraw in range(nsim):
            V = V_values[j]
            null_prob = null_prob_values[j]
            X, y, beta, sigma, true_signals, false_signals, detection_threshold = generate_signals(inpath, V=V, null_prob =null_prob)

            n, p = X.shape

            if n > 2 * p:
                dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
                sigma_ = np.sqrt(dispersion)
            else:
                sigma_ = np.std(y)

            lam_theory = sigma_ * lam_frac_values[j] * np.mean(
                np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

            output_split = (split_inf(X=X,
                                      y=y,
                                      beta=beta,
                                      true_signals=true_signals,
                                      false_signals=false_signals,
                                      detection_threshold=detection_threshold,
                                      split_proportion=0.50,
                                      lam_theory=lam_theory,
                                      sigma_=sigma_)).T

            df_split_1 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                  'true_screen', 'power_screen', 'false_screen',
                                                                  'power_total', 'false_total', 'power_selective',
                                                                  'ndiscoveries', 'true_dtotal', 'true_total'])

            df_split_1 = df_split_1.assign(simulation=ndraw,
                                           method="split (50%)",
                                           snr=V,
                                           null_probab=null_prob,
                                           runtime=0)

            output_split = (split_inf(X=X,
                                      y=y,
                                      beta=beta,
                                      true_signals=true_signals,
                                      false_signals=false_signals,
                                      detection_threshold=detection_threshold,
                                      split_proportion=0.60,
                                      lam_theory=lam_theory,
                                      sigma_=sigma_)).T

            df_split_2 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                  'true_screen', 'power_screen', 'false_screen',
                                                                  'power_total', 'false_total', 'power_selective',
                                                                  'ndiscoveries', 'true_dtotal', 'true_total'])

            df_split_2 = df_split_2.assign(simulation=ndraw,
                                           method="split (60%)",
                                           snr=V,
                                           null_probab=null_prob,
                                           runtime=0)

            output_split = (split_inf(X=X,
                                      y=y,
                                      beta=beta,
                                      true_signals=true_signals,
                                      false_signals=false_signals,
                                      detection_threshold=detection_threshold,
                                      split_proportion=0.70,
                                      lam_theory=lam_theory,
                                      sigma_=sigma_)).T

            df_split_3 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                  'true_screen', 'power_screen', 'false_screen',
                                                                  'power_total', 'false_total', 'power_selective',
                                                                  'ndiscoveries', 'true_dtotal', 'true_total'])

            df_split_3 = df_split_3.assign(simulation=ndraw,
                                           method="split (70%)",
                                           snr=V,
                                           null_probab=null_prob,
                                           runtime=0)

            output_split = (split_inf(X=X,
                                      y=y,
                                      beta=beta,
                                      true_signals=true_signals,
                                      false_signals=false_signals,
                                      detection_threshold=detection_threshold,
                                      split_proportion=0.80,
                                      lam_theory=lam_theory,
                                      sigma_=sigma_)).T

            df_split_4 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                  'true_screen', 'power_screen', 'false_screen',
                                                                  'power_total', 'false_total', 'power_selective',
                                                                  'ndiscoveries', 'true_dtotal', 'true_total'])

            df_split_4 = df_split_4.assign(simulation=ndraw,
                                           method="split (80%)",
                                           snr=V,
                                           null_probab=null_prob,
                                           runtime=0)

            output_split = (split_inf(X=X,
                                      y=y,
                                      beta=beta,
                                      true_signals=true_signals,
                                      false_signals=false_signals,
                                      detection_threshold=detection_threshold,
                                      split_proportion=0.90,
                                      lam_theory=lam_theory,
                                      sigma_=sigma_)).T

            df_split_5 = pd.DataFrame(data=output_split, columns=['coverage', 'length', 'nactive',
                                                                  'true_screen', 'power_screen', 'false_screen',
                                                                  'power_total', 'false_total', 'power_selective',
                                                                  'ndiscoveries', 'true_dtotal', 'true_total'])

            df_split_5 = df_split_5.assign(simulation=ndraw,
                                           method="split (90%)",
                                           snr=V,
                                           null_probab=null_prob,
                                           runtime=0)

            df_master = df_master.append(df_split_1, ignore_index=True)
            df_master = df_master.append(df_split_2, ignore_index=True)
            df_master = df_master.append(df_split_3, ignore_index=True)
            df_master = df_master.append(df_split_4, ignore_index=True)
            df_master = df_master.append(df_split_5, ignore_index=True)

            print("iteration completed ", ndraw + 1)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "realX_low_PF" + "_inference_35_split_90_" + target + ".csv")
    outfile_inf_html = os.path.join(outpath, "realX_low_PF" + "_inference_35_split_90_" + target + ".html")
    df_master.to_csv(outfile_inf_csv, index=False)
    df_master.to_html(outfile_inf_html)


def create_naive_output(V_values,
                        nsim,
                        inpath,
                        null_prob_values,
                        lam_frac_values,
                        target="selected",
                        outpath = None):

    df_master = pd.DataFrame()

    for j in range(V_values.shape[0]):
        for ndraw in range(nsim):
            V = V_values[j]
            null_prob = null_prob_values[j]
            X, y, beta, sigma, true_signals, false_signals, detection_threshold = generate_signals(inpath, V=V, null_prob =null_prob)
            print("check ", V, max(beta), beta[beta>1.].sum(), beta[beta>2.].sum())
            n, p = X.shape

            if n > 2 * p:
                dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
                sigma_ = np.sqrt(dispersion)
            else:
                sigma_ = np.std(y)/np.sqrt(2)

            lam_theory = sigma * lam_frac_values[j] * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

            output_naive = (naive_inf(X=X,
                                      y=y,
                                      beta=beta,
                                      true_signals=true_signals,
                                      false_signals=false_signals,
                                      detection_threshold=detection_threshold,
                                      lam_theory=lam_theory,
                                      sigma_=sigma)).T

            df_naive = pd.DataFrame(data=output_naive, columns=['coverage', 'length', 'nactive',
                                                                'true_screen', 'power_screen', 'false_screen',
                                                                'power_total', 'false_total', 'power_selective',
                                                                'ndiscoveries', 'true_dtotal', 'true_total'])

            df_naive = df_naive.assign(simulation=ndraw,
                                       method="naive",
                                       snr=V,
                                       null_probab=null_prob,
                                       runtime=0)

            df_master = df_master.append(df_naive, ignore_index=True)

            print("iteration completed ", ndraw + 1)

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "realX_PF" + "_inference_35_naive_90_" + target + ".csv")
    outfile_inf_html = os.path.join(outpath, "realX_PF" + "_inference_35_naive_90_" + target + ".html")
    df_master.to_csv(outfile_inf_csv, index=False)
    df_master.to_html(outfile_inf_html)

