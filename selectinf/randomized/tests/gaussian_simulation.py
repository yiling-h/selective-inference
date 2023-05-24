import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing
# from multiprocess import Pool
import sys

import regreg.api as rr

from selectinf.randomized.lasso import (lasso, split_lasso)
from selectinf.randomized.group_lasso_query import (group_lasso,
                                 split_group_lasso)

from selectinf.base import (full_targets,
                     selected_targets,
                     debiased_targets)
from selectinf.randomized.tests.instance import (gaussian_group_instance,
                                                 logistic_group_instance)

from selectinf.base import restricted_estimator
import scipy.stats

from selectinf.randomized.tests.test_group_lasso_simulation import (calculate_F1_score,
                                                                  naive_inference,
                                                                  randomization_inference,
                                                                  randomization_inference_fast,
                                                                  split_inference,
                                                                    data_splitting,
                                                                    posterior_inference)


def comparison_gaussian_lasso_vary_s(n=500,
                                     p=200,
                                          signal_fac=0.1,
                                          s=5,
                                          sigma=2,
                                          rho=0.3,
                                          randomizer_scale=1.,
                                          full_dispersion=True,
                                          level=0.90,
                                          range=range(0,100)):
    """
    Compare to R randomized lasso
    """

    # Operating characteristics
    oper_char = {}
    oper_char["sparsity size"] = []
    oper_char["coverage rate"] = []
    oper_char["avg length"] = []
    oper_char["method"] = []
    oper_char["F1 score"] = []
    # oper_char["runtime"] = []

    confint_df = pd.DataFrame()

    for s in [5, 8, 10]:  # [0.01, 0.03, 0.06, 0.1]:
        for i in range:
            np.random.seed(i)

            inst, const, const_split = gaussian_group_instance, group_lasso.gaussian, \
                                       split_group_lasso.gaussian
            signal = np.sqrt(signal_fac * 2 * np.log(p))
            signal_str = str(np.round(signal, decimals=2))

            while True:  # run until we get some selection
                groups = np.arange(50).repeat(4)
                X, Y, beta = inst(n=n,
                                  p=p,
                                  signal=signal,
                                  sgroup=s,
                                  groups=groups,
                                  ndiscrete=20,
                                  nlevels=5,
                                  sdiscrete=s - 3,  # s-3, # How many discrete rvs are not null
                                  equicorrelated=False,
                                  rho=rho,
                                  random_signs=True)[:3]
                # print(X)

                n, p = X.shape
                noselection = False  # flag for a certain method having an empty selected set

                if not noselection:
                    # MLE inference
                    start = time.perf_counter()
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.67,
                                                     beta=beta, groups=groups)
                    end = time.perf_counter()
                    MLE_runtime = end - start
                    # print(MLE_runtime)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds, nonzero_ds, beta_target_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups,
                                       proportion=0.67, level=0.9)
                    noselection = (coverage_ds is None)

                if not noselection:
                    # naive inference
                    coverage_naive, lengths_naive, nonzero_naive, conf_low_naive, conf_up_naive, \
                    beta_target_naive = \
                        naive_inference(X=X, Y=Y, groups=groups,
                                        beta=beta, const=const,
                                        n=n, level=level)
                    noselection = (coverage_naive is None)

                if not noselection:
                    # Posterior inference
                    coverage_pos, length_pos, beta_target_pos, nonzero_pos, conf_low_pos, conf_up_pos = \
                        posterior_inference(X=X, Y=Y, n=n, p=p, beta=beta, groups=groups)
                    noselection = (coverage_pos is None)

                if not noselection:
                    # F1 scores
                    # F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_ds)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)
                    F1_pos = calculate_F1_score(beta, selection=nonzero_pos)

                    # MLE coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage))
                    oper_char["avg length"].append(np.mean(length))
                    oper_char["F1 score"].append(F1)
                    oper_char["method"].append('MLE')
                    df_MLE = pd.concat([pd.DataFrame(np.ones(nonzero.sum()) * i),
                                        pd.DataFrame(beta_target),
                                        pd.DataFrame(conf_low),
                                        pd.DataFrame(conf_up),
                                        pd.DataFrame(beta[nonzero] != 0),
                                        pd.DataFrame(np.ones(nonzero.sum()) * s),
                                        pd.DataFrame(np.ones(nonzero.sum()) * F1),
                                        pd.DataFrame(["MLE"] * nonzero.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_MLE], axis=0)

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    # oper_char["runtime"].append(0)
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_ds.sum()) * i),
                                       pd.DataFrame(beta_target_ds),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_ds] != 0),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * s),
                                       pd.DataFrame(np.ones(nonzero_ds.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_ds.sum())
                                       ], axis=1)
                    confint_df = pd.concat([confint_df, df_ds], axis=0)

                    # Naive coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_naive))
                    oper_char["avg length"].append(np.mean(lengths_naive))
                    oper_char["F1 score"].append(F1_naive)
                    oper_char["method"].append('Naive')
                    df_naive = pd.concat([pd.DataFrame(np.ones(nonzero_naive.sum()) * i),
                                          pd.DataFrame(beta_target_naive),
                                          pd.DataFrame(conf_low_naive),
                                          pd.DataFrame(conf_up_naive),
                                          pd.DataFrame(beta[nonzero_naive] != 0),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * s),
                                          pd.DataFrame(np.ones(nonzero_naive.sum()) * F1_naive),
                                          pd.DataFrame(["Naive"] * nonzero_naive.sum())
                                          ], axis=1)
                    confint_df = pd.concat([confint_df, df_naive], axis=0)

                    # Posterior coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_pos))
                    oper_char["avg length"].append(np.mean(length_pos))
                    oper_char["F1 score"].append(F1_pos)
                    oper_char["method"].append('Posterior')
                    df_pos = pd.concat([pd.DataFrame(np.ones(nonzero_pos.sum()) * i),
                                        pd.DataFrame(beta_target_pos),
                                        pd.DataFrame(conf_low_pos),
                                        pd.DataFrame(conf_up_pos),
                                        pd.DataFrame(beta[nonzero_pos] != 0),
                                        pd.DataFrame(np.ones(nonzero_pos.sum()) * s),
                                        pd.DataFrame(np.ones(nonzero_pos.sum()) * F1_pos),
                                        pd.DataFrame(["Posterior"] * nonzero_pos.sum())
                                        ], axis=1)
                    confint_df = pd.concat([confint_df, df_pos], axis=0)

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    oper_char_df.to_csv('gaussian_vary_sparsity' + str(range.start) + '_' + str(range.stop) + '.csv', index=False)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames
    confint_df.to_csv('gaussian_CI_vary_sparsity' + str(range.start) + '_' + str(range.stop) + '.csv', index=False)


if __name__ == '__main__':
    argv = sys.argv
    start, end = int(argv[1]), int(argv[2])
    print("start:", start, ", end:", end)
    comparison_gaussian_lasso_vary_s(range=range(start, end))


