import numpy as np, os, itertools
import pandas as pd

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

from selection.adjusted_MLE.BH_MLE import (BHfilter,
                                           coverage,
                                           test_BH)

def output_file(n=500, p=500, rho=0.35, nsignals=np.array([10, 20, 30, 40, 50]), sigma =3.,
                target="marginal", randomizing_scale = np.sqrt(0.50), ndraw = 500, outpath = None,
                plot=False):


    df_selective_inference = pd.DataFrame()
    s_list = []

    for s in nsignals:
        s_list.append(s * np.ones(3))

        output_overall = np.zeros(23)
        for i in range(ndraw):
            output_overall += np.squeeze(
                test_BH(p=p, s=s, sigma=sigma, rho=rho, randomizer_scale=randomizing_scale, target=target, level=0.9,
                        q=0.1))

        nnaive = output_overall[22]
        nMLE = output_overall[21]

        randomized_MLE_inf = (output_overall[:7] / float(ndraw - nMLE)).reshape((1, 7))
        naive_inf = (output_overall[7:14] / float(ndraw - nnaive)).reshape((1, 7))
        fcr_inf = (output_overall[14:21] / float(ndraw - nnaive)).reshape((1, 7))

        df_naive = pd.DataFrame(data=naive_inf, columns=['coverage', 'length', 'tot-active', 'bias', 'sel-power', 'sel-fdr', 'tot-discoveries'])
        df_naive['method'] = "Naive"

        df_MLE = pd.DataFrame(data=randomized_MLE_inf, columns=['coverage', 'length', 'tot-active', 'bias', 'sel-power', 'sel-fdr','tot-discoveries'])
        df_MLE['method'] = "MLE"

        df_fcr = pd.DataFrame(data=fcr_inf, columns=['coverage', 'length', 'tot-active', 'bias', 'sel-power', 'sel-fdr', 'tot-discoveries'])
        df_fcr['method'] = "FCR"

        df_selective_inference = df_selective_inference.append(df_naive, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_MLE, ignore_index=True)
        df_selective_inference = df_selective_inference.append(df_fcr, ignore_index=True)

    s_list = list(itertools.chain.from_iterable(s_list))
    df_selective_inference['n'] = n
    df_selective_inference['p'] = p
    df_selective_inference['rho'] = rho
    df_selective_inference['beta-type'] = 'diff_strengths'
    df_selective_inference['s'] = pd.Series(np.asarray(s_list))
    df_selective_inference['target'] = target

    if outpath is None:
        outpath = os.path.dirname(__file__)

    outfile_inf_csv = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_" + target + "_rho_" + str(rho) + ".csv")
    outfile_inf_html = os.path.join(outpath, "dims_" + str(n) + "_" + str(p) + "_inference_" + target + "_rho_" + str(rho) + ".html")
    df_selective_inference.to_csv(outfile_inf_csv, index=False)
    df_selective_inference.to_html(outfile_inf_html)

output_file()


















