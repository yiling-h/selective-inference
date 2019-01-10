import numpy as np, sys, os, time, itertools
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

from scipy.stats import norm as ndist
from selection.randomized.lasso import lasso, full_targets, selected_targets, debiased_targets

def coverage(intervals, pval, target, truth):
    pval_alt = (pval[truth != 0]) < 0.1
    if pval_alt.sum() > 0:
        avg_power = np.mean(pval_alt)
    else:
        avg_power = 0.
    return np.mean((target > intervals[:, 0]) * (target < intervals[:, 1])), avg_power


def BHfilter(pval, q=0.2):
    robjects.r.assign('pval', pval)
    robjects.r.assign('q', q)
    robjects.r('Pval = p.adjust(pval, method="BH")')
    robjects.r('S = which((Pval < q)) - 1')
    S = robjects.r('S')
    ind = np.zeros(pval.shape[0], np.bool)
    ind[np.asarray(S, np.int)] = 1
    return ind


def sim_xy(n, p, nval, rho=0, s=5, beta_type=2, snr=1):
    robjects.r('''
    #library(bestsubset)
    source('~/best-subset/bestsubset/R/sim.R')
    sim_xy = sim.xy
    ''')

    r_simulate = robjects.globalenv['sim_xy']
    sim = r_simulate(n, p, nval, rho, s, beta_type, snr)
    X = np.array(sim.rx2('x'))
    y = np.array(sim.rx2('y'))
    X_val = np.array(sim.rx2('xval'))
    y_val = np.array(sim.rx2('yval'))
    Sigma = np.array(sim.rx2('Sigma'))
    beta = np.array(sim.rx2('beta'))
    sigma = np.array(sim.rx2('sigma'))

    return X, y, X_val, y_val, Sigma, beta, sigma

def compare_methods(n=500, p=100, nval=500, rho=0.35, s=5, beta_type=1, snr=0.20, target= "selected",
                    randomizer_scale=np.sqrt(0.50), full_dispersion=True, tuning_rand="lambda.theory"):

    X, y, _, _, Sigma, beta, sigma = sim_xy(n=n, p=p, nval=nval, rho=rho, s=s, beta_type=beta_type, snr=snr)
    print("snr", snr)
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n / (n - 1.)))
    y = y - y.mean()
    true_set = np.asarray([u for u in range(p) if beta[u] != 0])

    if full_dispersion:
        dispersion = np.linalg.norm(y - X.dot(np.linalg.pinv(X).dot(y))) ** 2 / (n - p)
        sigma_ = np.sqrt(dispersion)
    else:
        dispersion = None
        sigma_ = np.std(y)
    print("estimated and true sigma", sigma, sigma_)

    lam_theory = sigma_ * 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    randomized_lasso = lasso.gaussian(X,
                                      y,
                                      feature_weights=lam_theory * np.ones(p),
                                      randomizer_scale=np.sqrt(n) * randomizer_scale * sigma_)

    signs = randomized_lasso.fit()
    nonzero = signs != 0
    sys.stderr.write("active variables selected by randomized LASSO " + str(nonzero.sum()) + "\n" + "\n")
    active_set_rand = np.asarray([t for t in range(p) if nonzero[t]])
    active_rand_bool = np.asarray([(np.in1d(active_set_rand[x], true_set).sum() > 0) for x in range(nonzero.sum())],
                                  np.bool)
    nreport = 0.
    if nonzero.sum() > 0:
        if target == "full":
            target_randomized = beta[nonzero]
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = full_targets(randomized_lasso.loglike,
                                          randomized_lasso._W,
                                          nonzero,
                                          dispersion=dispersion)
        elif target == "selected":
            target_randomized = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))
            (observed_target,
             cov_target,
             cov_target_score,
             alternatives) = selected_targets(randomized_lasso.loglike,
                                              randomized_lasso._W,
                                              nonzero,
                                              dispersion=dispersion)
        else:
            raise ValueError('not a valid specification of target')
        toc = time.time()
        MLE_estimate, _, _, MLE_pval, MLE_intervals, ind_unbiased_estimator = randomized_lasso.selective_MLE(observed_target,
                                                                                                             cov_target,
                                                                                                             cov_target_score,
                                                                                                             alternatives)
        tic = time.time()
        time_MLE = tic - toc

        cov_MLE, selective_MLE_power = coverage(MLE_intervals, MLE_pval, target_randomized, beta[nonzero])
        length_MLE = np.mean(MLE_intervals[:, 1] - MLE_intervals[:, 0])
        power_MLE = ((active_rand_bool) * (
            np.logical_or((0. < MLE_intervals[:, 0]), (0. > MLE_intervals[:, 1])))).sum() / float((beta != 0).sum())
        MLE_discoveries = BHfilter(MLE_pval, q=0.1)
        power_MLE_BH = (MLE_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_MLE_BH = (MLE_discoveries * ~active_rand_bool).sum() / float(max(MLE_discoveries.sum(), 1.))
        bias_MLE = np.mean(MLE_estimate - target_randomized)

        toc = time.time()
        intervals_uni, pvalue_uni = randomized_lasso.inference_new(observed_target,
                                                           cov_target,
                                                           cov_target_score,
                                                           alternatives)

        tic = time.time()
        time_uni = tic - toc
        intervals_uni = intervals_uni.T
        cov_uni, selective_uni_power = coverage(intervals_uni, pvalue_uni, target_randomized,beta[nonzero])
        length_uni = np.mean(intervals_uni[:, 1] - intervals_uni[:, 0])
        power_uni = ((active_rand_bool) * (np.logical_or((0. < intervals_uni[:, 0]),
                                                             (0. > intervals_uni[:, 1])))).sum() / float((beta != 0).sum())
        uni_discoveries = BHfilter(pvalue_uni, q=0.1)
        power_uni_BH = (uni_discoveries * active_rand_bool).sum() / float((beta != 0).sum())
        fdr_uni_BH = (uni_discoveries * ~active_rand_bool).sum() / float(max(uni_discoveries.sum(), 1.))
        bias_randLASSO = np.mean(randomized_lasso.initial_soln[nonzero] - target_randomized)

    else:
        nreport += 1
        cov_MLE, length_MLE, power_MLE, power_MLE_BH, fdr_MLE_BH, bias_MLE, selective_MLE_power, time_MLE = [0., 0., 0.,
                                                                                                             0., 0., 0.,
                                                                                                             0., 0.]
        cov_uni, length_uni, power_uni, power_uni_BH, fdr_uni_BH, bias_randLASSO, selective_uni_power, time_uni = [0., 0., 0.,
                                                                                                                   0., 0., 0.,
                                                                                                                   0., 0.]
        MLE_discoveries = np.zeros(1)
        uni_discoveries = np.zeros(1)

    MLE_inf = np.vstack((cov_MLE, length_MLE, 0., nonzero.sum(), bias_MLE, selective_MLE_power, time_MLE,
                         power_MLE, power_MLE_BH, fdr_MLE_BH, MLE_discoveries.sum()))

    uni_inf = np.vstack((cov_uni, length_uni, 0., nonzero.sum(), bias_randLASSO,
                         selective_uni_power, time_uni, power_uni, power_uni_BH,
                         fdr_uni_BH, uni_discoveries.sum()))

    return np.vstack((MLE_inf, uni_inf, nreport))

def output_compare_sampler_mle(n=500, p=100, rho=0.35, s=5, beta_type=1, snr_values=np.array([0.10, 0.15, 0.20, 0.30]),
                               target="full", tuning_rand="lambda.theory", randomizing_scale = np.sqrt(0.50), ndraw = 20, outpath = None):

    df_selective_inference = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(2))
        output_overall = np.zeros(23)
        for i in range(ndraw):
            output_overall += np.squeeze(
                compare_methods(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr, target = target,
                                randomizer_scale=randomizing_scale, full_dispersion=full_dispersion, tuning_rand=tuning_rand))

        nreport = output_overall[22]
        randomized_MLE_inf = np.hstack(((output_overall[0:7] / float(ndraw - nreport)).reshape((1, 7)),
                                       (output_overall[7:11] / float(ndraw)).reshape((1, 4))))
        randomized_sampler_inf = np.hstack(((output_overall[11:18] / float(ndraw - nreport)).reshape((1, 7)),
                                        (output_overall[18:22] / float(ndraw)).reshape((1, 4))))

        df_MLE = pd.DataFrame(data=randomized_MLE_inf, columns=['coverage', 'length', 'prop-infty', 'tot-active','bias', 'sel-power', 'time',
                                                                'power', 'power-BH', 'fdr-BH', 'tot-discoveries'])
        df_MLE['method'] = "MLE"
        df_sampler = pd.DataFrame(data=randomized_sampler_inf, columns=['coverage', 'length', 'prop-infty', 'tot-active', 'bias', 'sel-power', 'time',
                                                                        'power', 'power-BH', 'fdr-BH', 'tot-discoveries'])
        df_sampler['method'] = "More-conditioning"

        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_sampler, ignore_index=True)

    snr_list = list(itertools.chain.from_iterable(snr_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['s'] = s
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = beta_type
    df_selective_inference['snr'] = pd.Series(np.asarray(snr_list))
    df_selective_inference['target'] = target

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "compare_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "compare_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)

output_compare_sampler_mle(outpath='/Users/psnigdha/adjusted_MLE/new_method/')

