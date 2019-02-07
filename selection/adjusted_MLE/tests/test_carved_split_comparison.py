import numpy as np, os, itertools
import pandas as pd

from selection.adjusted_MLE.cv_MLE import (sim_xy,
                                           selInf_R,
                                           glmnet_lasso,
                                           BHfilter,
                                           coverage,
                                           compare_split_MLE)

def output_compare_carved_split(n=500, p=100, rho=0.35, s=5, beta_type=1, snr_values=np.array([0.15, 0.21, 0.26, 0.31, 0.36,
                                                                                              0.42, 0.71, 1.22, 2.07, 3.52]),
                               target="selected", tuning_rand="lambda.theory", tuning_nonrand ="lambda.theory",
                               split_proportion = 0.67, ndraw = 50, outpath = None):

    df_selective_inference = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(2))
        output_overall = np.zeros(26)
        for i in range(ndraw):
            output_overall += np.squeeze(compare_split_MLE(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                           target = target, full_dispersion=full_dispersion, tuning_rand=tuning_rand,
                                                           tuning_nonrand=tuning_nonrand,split_proportion=split_proportion))

        nreport = output_overall[24]
        nreport_split = output_overall[25]
        carved_inf = np.hstack(((output_overall[0:8] / float(ndraw - nreport)).reshape((1, 8)),
                                (output_overall[8:12] / float(ndraw)).reshape((1, 4))))
        split_inf = np.hstack(((output_overall[12:20] / float(ndraw - nreport_split)).reshape((1, 8)),
                               (output_overall[20:24] / float(ndraw)).reshape((1, 4))))

        df_MLE = pd.DataFrame(data=carved_inf, columns=['coverage', 'length', 'prop-infty', 'tot-active','bias', 'sel-power', 'time','fdr-inf',
                                                                'power', 'power-BH', 'fdr-BH', 'tot-discoveries'])
        df_MLE['method'] = "Carved"
        df_split = pd.DataFrame(data=split_inf, columns=['coverage', 'length', 'prop-infty', 'tot-active', 'bias', 'sel-power', 'time','fdr-inf',
                                                                        'power', 'power-BH', 'fdr-BH', 'tot-discoveries'])
        df_split['method'] = "Split"

        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_split, ignore_index=True)

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

    outfile_inf_csv = os.path.join(outpath, "splitcarved_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "splitcarved_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)

output_compare_carved_split(outpath='/Users/psnigdha/adjusted_MLE/n_500_p_100/lam_theory/betatype_1/')
