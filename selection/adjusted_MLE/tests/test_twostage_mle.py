import numpy as np, os, itertools
import pandas as pd

from selection.adjusted_MLE.twostage_mle import (sim_xy,
                                                 compare_twostage_mle)


def output_compare_sampler_mle(n=3000, p=1000, rho=0.35, s=20, beta_type=1, snr_values=np.array([0.15, 0.21, 0.26, 0.31, 0.36,
                                                                                                 0.42, 0.71, 1.22, 2.07, 3.52]),
                               randomizing_scale = np.sqrt(0.50), ndraw = 50, outpath = None):

    df_selective_inference = pd.DataFrame()

    if n > p:
        full_dispersion = True
    else:
        full_dispersion = False

    snr_list = []
    for snr in snr_values:
        snr_list.append(snr*np.ones(2))
        output_overall = np.zeros(14)
        for i in range(ndraw):
            output_overall += np.squeeze(compare_twostage_mle(n=n, p=p, nval=n, rho=rho, s=s, beta_type=beta_type, snr=snr,
                                                              randomizer_scale=randomizing_scale, full_dispersion=full_dispersion))

        nreport = output_overall[12]
        nreport_nonrand = output_overall[13]
        randomized_MLE_inf = np.hstack(((output_overall[0:4] / float(ndraw - nreport)).reshape((1, 4)),
                                        (output_overall[4:6] / float(ndraw)).reshape((1, 2))))
        Naive_inf = np.hstack(((output_overall[6:10] / float(ndraw - nreport_nonrand)).reshape((1, 4)),
                               (output_overall[10:12] / float(ndraw)).reshape((1, 2))))

        df_MLE = pd.DataFrame(data= randomized_MLE_inf, columns=['coverage', 'length', 'sel-power', 'fdr', 'nactive-ms', 'nactive-slope'])
        df_MLE['method'] = "MLE"
        df_naive = pd.DataFrame(data= Naive_inf, columns=['coverage', 'length', 'sel-power', 'fdr', 'nactive-ms', 'nactive-slope'])
        df_naive['method'] = "Naive"

        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_naive, ignore_index=True)

    snr_list = list(itertools.chain.from_iterable(snr_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['s'] = s
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = beta_type
    df_selective_inference['snr'] = pd.Series(np.asarray(snr_list))

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "twostage_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "twostage_" + str(n) + "_" + str(p) + "_inference_betatype" + str(beta_type) + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)

output_compare_sampler_mle(outpath='/Users/psnigdha/adjusted_MLE/twostage_mle/')


