import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt

import regreg.api as rr

from ..lasso import (lasso,
                     split_lasso)
from ..group_lasso_query import (group_lasso)
from ..group_lasso_query_quasi import (group_lasso_quasi)

from ...base import (full_targets,
                     selected_targets,
                     debiased_targets,
                     selected_targets_quasi)
from selectinf.randomized.tests.instance import (gaussian_group_instance,
                                                 logistic_group_instance,
                                                 poisson_group_instance,
                                                 quasi_poisson_group_instance)
from selectinf.tests.instance import (gaussian_instance,
                               logistic_instance,
                               poisson_instance,
                               cox_instance)
from ...base import restricted_estimator

def test_possion_vs_quasi_solutions(n=500,
                                    p=200,
                                    signal_fac=0.1,  # 1.2
                                    sgroup=5,
                                    groups=np.arange(50).repeat(4),
                                    rho=0.3,
                                    weight_frac=1.,
                                    randomizer_scale=1,
                                    level=0.90,
                                    iter=50):
        # Operating characteristics
        oper_char = {}
        oper_char["coverage rate"] = []
        oper_char["avg length"] = []
        oper_char["sparsity size"] = []
        oper_char["method"] = []

        for s in [2, 5, 8, 10]:
            for i in range(iter):

                inst = quasi_poisson_group_instance
                signal = np.sqrt(signal_fac * 2 * np.log(p))
                print("signal:", signal)

                while True:  # run until we get some selection
                    X, Y, beta = inst(n=n,
                                      p=p,
                                      signal=signal,
                                      sgroup=sgroup,
                                      groups=groups,
                                      ndiscrete=0,
                                      sdiscrete=0,
                                      equicorrelated=False,
                                      rho=rho,
                                      phi=1.5,
                                      random_signs=True, # changed
                                      center=False,
                                      scale=True)[:3]     # changed
                    print("X mean:", np.mean(X), "X var", np.var(X))
                    print("beta norm", np.linalg.norm(beta))
                    print("Y mean:", np.mean(Y))
                    n, p = X.shape

                    ##estimate noise level in data

                    sigma_ = np.std(Y)
                    print(sigma_)

                    ##solve group LASSO with group penalty weights = weights

                    weights = dict([(i, weight_frac * sigma_ * np.sqrt(2 * np.log(p))) for i in np.unique(groups)])


                    conv = group_lasso.poisson(X=X,
                                               counts=Y,
                                               groups=groups,
                                               weights=weights,
                                               useJacobian=True,
                                               ridge_term=0.,
                                               randomizer_scale=randomizer_scale * sigma_)


                    signs_poisson, _ = conv.fit()
                    nonzero_poisson = (signs_poisson != 0)
                    #print(nonzero_poisson.shape)

                    omega = conv._initial_omega

                    conv_quasi = group_lasso_quasi.quasipoisson(X=X,
                                                                counts=Y,
                                                                groups=groups,
                                                                weights=weights,
                                                                useJacobian=True,
                                                                ridge_term=0.,
                                                                perturb=omega,
                                                                randomizer_scale=randomizer_scale * sigma_)


                    signs_quasi, _ = conv_quasi.fit()
                    nonzero_quasi = (signs_quasi != 0)

                    if nonzero_quasi.sum() > 0 and nonzero_poisson.sum() > 0:
                        # Solving the inferential target
                        def solve_target_restricted():
                            Y_mean = np.exp(X.dot(beta))

                            loglike = rr.glm.poisson(X, counts=Y_mean)
                            # For LASSO, this is the OLS solution on X_{E,U}
                            _beta_unpenalized = restricted_estimator(loglike,
                                                                     nonzero_quasi)
                            return _beta_unpenalized

                        # Quasi-poisson inference
                        conv_quasi.setup_inference(dispersion=1)

                        #cov_score_quasi = conv_quasi._unscaled_cov_score
                        cov_score_quasi = conv_quasi._unscaled_cov_score

                        target_spec_quasi = selected_targets_quasi(loglike=conv_quasi.loglike,
                                                                   solution=conv_quasi.observed_soln,
                                                                   cov_score=cov_score_quasi,
                                                                   dispersion=1)


                        # Poisson inference
                        conv.setup_inference(dispersion=1)

                        target_spec = selected_targets(loglike=conv.loglike,
                                                       solution=conv.observed_soln,
                                                       dispersion=1)
                        result_quasi = conv_quasi.inference(target_spec_quasi,
                                                            method='selective_MLE',
                                                            level=level)
                        result = conv.inference(target_spec,
                                                method='selective_MLE',
                                                level=level)

                        # Comparing results
                        pval = result['pvalue']
                        intervals = np.asarray(result[['lower_confidence', 'upper_confidence']])
                        pval_quasi = result_quasi['pvalue']
                        intervals_quasi = np.asarray(result_quasi[['lower_confidence', 'upper_confidence']])

                        beta_target = solve_target_restricted()
                        coverage = (beta_target > intervals[:, 0]) * (beta_target < intervals[:, 1])
                        coverage_quasi = (beta_target > intervals_quasi[:, 0]) * (beta_target < intervals_quasi[:, 1])

                        """
                        # Tabulate results
                        d = {'target': beta_target,
                             'L_Poisson': intervals[:, 0], 'U_Poisson': intervals[:, 1],
                             'Coverage_P': coverage,
                             'L_Quasi': intervals_quasi[:, 0], 'U_Quasi': intervals_quasi[:, 1],
                             'Coverage_Q': coverage_quasi}
                        df = pd.DataFrame(data=d)
                        print(df)
                        """

                        # MLE coverage
                        oper_char["coverage rate"].append(np.mean(coverage))
                        oper_char["sparsity size"].append(s)
                        oper_char["avg length"].append(np.mean(intervals[:, 1] - intervals[:, 0]))
                        oper_char["method"].append("Poisson")

                        oper_char["coverage rate"].append(np.mean(coverage_quasi))
                        oper_char["sparsity size"].append(s)
                        oper_char["avg length"].append(np.mean(intervals_quasi[:, 1] - intervals_quasi[:, 0]))
                        oper_char["method"].append("Quasi Poisson")

                        # print(np.round(intervals[:, 0],1))
                        # print(np.round(intervals[:, 1], 1))
                        # print(np.round(beta_target, 1))

                        break  # Go to next iteration if we have some selection
        oper_char_df = pd.DataFrame.from_dict(oper_char)
        oper_char_df.to_csv('selectinf/randomized/tests/oper_char_vary_s_qp_phi1_5.csv', index=False)

def test_plot_from_csv(path='selectinf/randomized/tests/oper_char_vary_s_qp_phi1_5.csv'):
    oper_char_df = pd.read_csv(path)
    cov_plot = sns.boxplot(y=oper_char_df["coverage rate"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    cov_plot.set_ylim(0.2, 1)
    cov_plot.set(title='Coverage (phi=1.5)')
    plt.show()

    len_plot = sns.boxplot(y=oper_char_df["avg length"],
                           x=oper_char_df["sparsity size"],
                           hue=oper_char_df["method"],
                           showmeans=True,
                           orient="v")
    len_plot.set_ylim(0, 15)
    len_plot.set(title='Length (phi=1.5)')
    plt.show()