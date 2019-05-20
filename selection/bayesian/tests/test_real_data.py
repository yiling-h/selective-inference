import numpy as np, itertools
from selection.randomized.lasso import lasso, selected_targets, full_targets, debiased_targets
from selection.bayesian.utils import glmnet_lasso, glmnet_lasso_cv1se, glmnet_lasso_cvmin, power_fdr
from selection.bayesian.generative_instance import generate_data, generate_data_instance
from selection.bayesian.posterior_lasso import inference_lasso
from scipy.stats import norm as ndist


def generate_signals(detection_frac = 0.44, false_frac= 0.20, sigma=1.):
    X = np.load("/Users/psnigdha/Research/RadioiBAG/Data/X.npy")
    n, p = X.shape

    clusters = np.load("/Users/psnigdha/Research/RadioiBAG/Data/clusters.npy").astype(int)
    cluster_size = (np.load("/Users/psnigdha/Research/RadioiBAG/Data/cluster_size.npy")).astype(int)

    X = X[:,clusters]
    X -= X.mean(0)[None, :]
    scalingX = (X.std(0)[None, :] * np.sqrt(n))
    X /= scalingX

    beta_true = np.zeros(p)
    position_bool = np.zeros(p, np.bool)
    detection_threshold = detection_frac * np.sqrt(2. * np.log(p))

    strong = []
    null = []
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.90:
            sig = np.random.laplace(loc=0., scale=0.10)
            if sig > detection_threshold:
                strong.append(sig)
            else:
                null.append(sig)
        else:
            sig = np.random.laplace(loc=0., scale=6.0)
            if sig > detection_threshold:
                strong.append(sig)
            else:
                null.append(sig)

    strong = np.asarray(strong)
    null = np.asarray(null)

    cluster_length = np.cumsum(cluster_size)
    cluster_choice = np.random.choice(cluster_size.shape[0], strong.shape[0],replace=False)
    true_clusters = []

    for j in range(strong.shape[0]):
        pos_wcluster = np.random.choice(cluster_size[cluster_choice[j]], 1)
        if cluster_choice[j]>0:
            beta_true[cluster_length[cluster_choice[j]-1]+pos_wcluster] = strong[j]
            position_bool[cluster_length[cluster_choice[j]-1]+pos_wcluster] = 1
            if strong[j]> detection_threshold:
                true_clusters.append(cluster_length[cluster_choice[j]-1] + np.arange(cluster_size[cluster_choice[j]]))
        else:
            beta_true[pos_wcluster] = strong[j]
            position_bool[pos_wcluster] = 1
            if strong[j] > detection_threshold:
                true_clusters.append(np.arange(cluster_size[cluster_choice[j]]))

    beta_true[~position_bool] = null
    Y = (X.dot(beta_true) + np.random.standard_normal(n)) * sigma

    cluster_list = []
    false_clusters = []
    for k in range(cluster_size.shape[0]):
        if k==0:
            clust_ind = clusters[:cluster_length[k]]
            cluster_list.append(clust_ind)
            if max(np.abs(beta_true[clust_ind])) < false_frac:
                false_clusters.append(clust_ind)
        else:
            clust_ind = clusters[cluster_length[k-1]:cluster_length[k]]
            cluster_list.append(clust_ind)
            if max(np.abs(beta_true[clust_ind])) < false_frac:
                false_clusters.append(clust_ind)

    return X, Y, beta_true * sigma, sigma, true_clusters, false_clusters, cluster_list, detection_threshold

#generate_signals()

def discoveries_count(active_set, signal_clusters, false_clusters, clusters):

    true_discoveries = 0.
    false_discoveries = 0.
    discoveries = 0.
    for i in range(len(signal_clusters)):
        inter = np.intersect1d(active_set, signal_clusters[i])
        if inter.shape[0]>0:
            true_discoveries += 1
    for j in range(len(clusters)):
        inter = np.intersect1d(active_set, clusters[j])
        if inter.shape[0]>0:
            discoveries += 1
    for k in range(len(false_clusters)):
        inter = np.intersect1d(active_set, false_clusters[k])
        if inter.shape[0] > 0:
            false_discoveries += 1

    return true_discoveries, false_discoveries, discoveries

def compare_inference(randomizer_scale= 1.,
                      split_proportion= 0.60,
                      target="selected",
                      lam_frac= 1.1):

    X, y, beta, sigma, true_clusters, false_clusters, clusters, detection_threshold = generate_signals()
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
    nreport = 0.
    nactive = nonzero.sum()

    if nonzero.sum() > 0 and nonzero.sum() < 30:

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
        true_screen, false_screen, tot_screen = discoveries_count(active_screenset, true_clusters, false_clusters, clusters)

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

        samples, count = posterior_inf.posterior_sampler(nsample=2200, nburnin=200, step=1., start=None,
                                                         Metropolis=False)

        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = np.mean((lci < beta_target) * (uci > beta_target))
        length = np.mean(uci - lci)

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
        fdr_selective = false_total
        ndiscoveries = reportind.sum()

    else:
        nreport = 1.
        coverage, length, power_screen, false_screen, power_total, false_total, \
        power_selective, fdr_selective, ndiscoveries, true_screen, true_dtotal = [0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                                                  0., 0.]

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
    lam_theory_split = sigma_ * lam_frac * np.mean(
        np.fabs(np.dot(X_sel_scaled.T, np.random.standard_normal((subsample_size, 2000)))).max(0))
    glm_LASSO_split = glmnet_lasso(X_sel_scaled, y_sel, lam_theory_split / float(subsample_size))
    active_LASSO_split = (glm_LASSO_split != 0)
    nactive_split = active_LASSO_split.sum()

    if nactive_split > 0 and nactive_split < 30:
        X_split = X[:, active_LASSO_split]
        active_screenset_split = np.asarray([s for s in range(p) if active_LASSO_split[s]])

        true_screen_split, false_screen_split, tot_screen_split = discoveries_count(active_screenset_split, true_clusters,
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
        except:
            projection_active_split = X_inf_split.dot(np.linalg.inv(X_inf_split.T.dot(X_inf_split) + 0.50))

        true_val_split = np.linalg.pinv(X_split).dot(X.dot(beta))

        est_split = projection_active_split.T.dot(y_inf)
        M_split = np.linalg.inv(prior_variance * np.identity(nactive_split) + noise_variance * np.linalg.inv(
            X_inf_split.T.dot(X_inf_split)))
        post_mean_split = prior_variance * (M_split).dot(est_split)
        post_var_split = prior_variance * np.identity(nactive_split) - (prior_variance ** 2.) * (M_split)
        adjusted_intervals_split = np.vstack([post_mean_split - 1.65 * (np.sqrt(np.diag(post_var_split))),
                                              post_mean_split + 1.65 * (np.sqrt(np.diag(post_var_split)))])

        coverage_split = np.mean(
            (true_val_split > adjusted_intervals_split[0, :]) * (true_val_split < adjusted_intervals_split[1, :]))
        length_split = np.mean(adjusted_intervals_split[1, :] - adjusted_intervals_split[0, :])

        reportind_split = np.zeros(nactive_split, np.bool)
        posterior_mass_pos = ndist.cdf((detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
        posterior_mass_neg = ndist.cdf((-detection_threshold - post_mean_split) / (np.sqrt(np.diag(post_var_split))))
        for u in range(nactive_split):
            if 1. - posterior_mass_pos[u] > 0.50 or posterior_mass_neg[u] > 0.50:
                reportind_split[u] = 1

        reportset_split = np.asarray(
            [active_screenset_split[f] for f in range(nactive_split) if reportind_split[f] == 1])

        true_dtotal_split, false_dtotal_split, dtotal_split = discoveries_count(reportset_split, true_clusters, false_clusters, clusters)
        power_total_split = true_dtotal_split / max(float(len(true_clusters)), 1.)
        false_total_split = false_dtotal_split / max(float(dtotal_split), 1.)

        power_selective_split = true_dtotal_split / max(float(true_screen_split), 1.)
        fdr_selective_split = false_total_split
        ndiscoveries_split = reportind_split.sum()

    else:
        nreport_split = 1.
        coverage_split, length_split, power_screen_split, false_screen_split, \
        power_total_split, false_total_split, power_selective_split, \
        fdr_selective_split, ndiscoveries_split, true_screen_split, true_dtotal_split = [0., 0., 0., 0., 0., 0., 0., 0.,
                                                                                         0., 0.]

    return np.vstack((coverage, length, nactive, true_screen, power_screen, false_screen,
                      power_total, false_total, power_selective, fdr_selective, ndiscoveries, true_dtotal,
                      coverage_split, length_split, nactive_split, true_screen_split, power_screen_split, false_screen_split,
                      power_total_split, false_total_split, power_selective_split, fdr_selective_split,
                      ndiscoveries_split, true_dtotal_split,
                      nreport, nreport_split))


def main(ndraw=10, split_proportion=0.70, randomizer_scale=1.):
    output = np.zeros(26)
    exception = 0.
    linalg_error = 0.
    for n in range(ndraw):
        try:
            output += np.squeeze(compare_inference(randomizer_scale=randomizer_scale,
                                                   split_proportion=split_proportion,
                                                   target="selected"))
        except ValueError:
               #"Singular matrix" in str(np.linalg.LinAlgError):
            if ValueError:
                exception += 1.
            else:
                linalg_error += 1.
            pass

        print("iteration completed ", n + 1)
        print("adjusted inferential metrics so far ", output[:12] / (n + 1. - output[24] - exception - linalg_error))
        print("split inferential metrics so far ", output[12:24] / (n + 1. - output[25] - exception - linalg_error))
        print("exceptions ", exception, linalg_error)


main(ndraw=50)
