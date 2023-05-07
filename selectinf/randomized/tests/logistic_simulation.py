import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing
# from multiprocess import Pool

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

from selectinf.randomized.tests.test_logistic_group_lasso import (calculate_F1_score,
                                                                  naive_inference,
                                                                  randomization_inference,
                                                                  randomization_inference_fast,
                                                                  split_inference, data_splitting)

def comparison_logistic(range):
    """
        Compare to R randomized lasso
        """
    n = 500
    p = 200
    signal_fac = 0.1
    sigma = 2
    rho = 0.5
    randomizer_scale = 1.
    full_dispersion = True
    level = 0.90

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

            inst, const, const_split = logistic_group_instance, group_lasso.logistic, \
                                       split_group_lasso.logistic
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
                    # carving
                    coverage_s, length_s, beta_target_s, nonzero_s, \
                    selection_idx_s, hessian, conf_low_s, conf_up_s = \
                        split_inference(X=X, Y=Y, n=n, p=p,
                                        beta=beta, groups=groups, const=const_split,
                                        proportion=0.67)

                    noselection = (coverage_s is None)

                if not noselection:
                    # MLE inference
                    start = time.perf_counter()
                    coverage, length, beta_target, nonzero, conf_low, conf_up = \
                        randomization_inference_fast(X=X, Y=Y, n=n, p=p, proportion=0.67,
                                                     beta=beta, groups=groups, hess=hessian)
                    end = time.perf_counter()
                    MLE_runtime = end - start
                    # print(MLE_runtime)
                    noselection = (coverage is None)

                if not noselection:
                    # data splitting
                    coverage_ds, lengths_ds, conf_low_ds, conf_up_ds = \
                        data_splitting(X=X, Y=Y, n=n, p=p, beta=beta, nonzero=nonzero_s,
                                       subset_select=selection_idx_s, level=0.9)
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
                    # F1 scores
                    F1_s = calculate_F1_score(beta, selection=nonzero_s)
                    F1 = calculate_F1_score(beta, selection=nonzero)
                    F1_ds = calculate_F1_score(beta, selection=nonzero_s)
                    F1_naive = calculate_F1_score(beta, selection=nonzero_naive)

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

                    # Carving coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_s))
                    oper_char["avg length"].append(np.mean(length_s))
                    oper_char["F1 score"].append(F1_s)
                    oper_char["method"].append('Carving')
                    # oper_char["runtime"].append(0)
                    df_s = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                      pd.DataFrame(beta_target_s),
                                      pd.DataFrame(conf_low_s),
                                      pd.DataFrame(conf_up_s),
                                      pd.DataFrame(beta[nonzero_s] != 0),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * s),
                                      pd.DataFrame(np.ones(nonzero_s.sum()) * F1_s),
                                      pd.DataFrame(["Carving"] * nonzero_s.sum())
                                      ], axis=1)
                    confint_df = pd.concat([confint_df, df_s], axis=0)

                    # Data splitting coverage
                    oper_char["sparsity size"].append(s)
                    oper_char["coverage rate"].append(np.mean(coverage_ds))
                    oper_char["avg length"].append(np.mean(lengths_ds))
                    oper_char["F1 score"].append(F1_ds)
                    oper_char["method"].append('Data splitting')
                    # oper_char["runtime"].append(0)
                    df_ds = pd.concat([pd.DataFrame(np.ones(nonzero_s.sum()) * i),
                                       pd.DataFrame(beta_target_s),
                                       pd.DataFrame(conf_low_ds),
                                       pd.DataFrame(conf_up_ds),
                                       pd.DataFrame(beta[nonzero_s] != 0),
                                       pd.DataFrame(np.ones(nonzero_s.sum()) * s),
                                       pd.DataFrame(np.ones(nonzero_s.sum()) * F1_ds),
                                       pd.DataFrame(["Data splitting"] * nonzero_s.sum())
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

                    break  # Go to next iteration if we have some selection

    oper_char_df = pd.DataFrame.from_dict(oper_char)
    colnames = ['Index'] + ['target'] + ['LCB'] + ['UCB'] + ['TP'] + ['sparsity size'] + ['F1'] + ['Method']
    confint_df.columns = colnames

    print("task done")
    return oper_char_df, confint_df

def test_comparison_logistic_group_lasso_vary_s_parallel(iter=20,
                                                         ncore=2):
    print(iter)
    print(ncore)
    def n_range_to_k(n, k):
        l = []
        for i in range(k):
            if i == 0:
                start = 0
                end = int(n / k)
            elif i == k - 1:
                start = end
                end = n
            else:
                start = end
                end = int((i + 1) * n / k)
            range_i = range(start, end)
            l.append(range_i)
        return l

    range_list = n_range_to_k(n=iter,k=ncore)

    print("ranges:", range_list)


    pool = multiprocessing.Pool(processes=ncore)
    pool_outputs = pool.map(comparison_logistic, range_list)

    # with Pool(ncore) as pool:
    #     pool_outputs = list(
    #         tqdm(
    #             pool.imap(comparison_logistic, range_list),
    #             total=len(range_list)
    #       )
    #  )

    oper_char_df = pd.DataFrame()
    confint_df = pd.DataFrame()

    for i in range(ncore):
        oper_char_df = pd.concat([oper_char_df, pool_outputs[i][0]], axis=0)
        confint_df = pd.concat([confint_df, pool_outputs[i][1]], axis=0)

    #oper_char_df.to_csv('selectinf/randomized/tests/logis_vary_sparsity.csv', index=False)
    #confint_df.to_csv('selectinf/randomized/tests/logis_CI_vary_sparsity.csv', index=False)
    oper_char_df.to_csv('selectinf/randomized/tests/logis_vary_sparsity.csv', index=False)
    confint_df.to_csv('selectinf/randomized/tests/logis_CI_vary_sparsity.csv', index=False)

    def print_results(oper_char_df):
        print("Mean coverage rate/length:")
        print(oper_char_df.groupby(['sparsity size', 'method']).mean())

        sns.boxplot(y=oper_char_df["coverage rate"],
                    x=oper_char_df["sparsity size"],
                    hue=oper_char_df["method"],
                    orient="v")
        plt.show()

        len_plot = sns.boxplot(y=oper_char_df["avg length"],
                               x=oper_char_df["sparsity size"],
                               hue=oper_char_df["method"],
                               showmeans=True,
                               orient="v")
        len_plot.set_ylim(5, 17)
        plt.show()

        F1_plot = sns.boxplot(y=oper_char_df["F1 score"],
                              x=oper_char_df["sparsity size"],
                              hue=oper_char_df["method"],
                              showmeans=True,
                              orient="v")
        F1_plot.set_ylim(0, 1)
        plt.show()

    #print_results(oper_char_df)


"""def test_save():
    mat = np.random.normal(size=(100,10))
    data = pd.DataFrame(mat)
    data.to_csv('test_save.csv', index=False)"""

if __name__ == '__main__':
    test_comparison_logistic_group_lasso_vary_s_parallel()
    # test_save()

