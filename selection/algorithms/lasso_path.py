"""
Selective tests along the LASSO path, as described in 
sequential testing paper
"""

import warnings

import numpy as np

import regreg.api as rr

from selection.algorithms.lasso import lasso
from selection.constraints.affine import (constraints, gibbs_test,
                                          stack as con_stack)

class lasso_path(lasso):

    def __init__(self, X, y, lambda_values, feature_weights=None):
        n, p = X.shape
        self.X = X
        self.y = y
        self.lambda_values = lambda_values
        self.ever_active = set([])

        self.loss = rr.squared_error(X, y)

        if feature_weights is None:
            feature_weights = np.ones(p)
        self.penalty = rr.weighted_l1norm(feature_weights, 
                                          lagrange=self.lambda_values[0])
        self.problem = rr.simple_problem(self.loss, self.penalty)
        self.soln_path = []

    def fit(self, test_statistic, burnin=5000, ndraw=20000,
            **solve_args):
        n, p = self.X.shape

        for idx, lam in enumerate(self.lambda_values):
            self.penalty.lagrange = lam
            soln = self.problem.solve(**solve_args)
            self.soln_path.append(soln.copy())

            active_set = np.nonzero(soln != 0)[0]
            diff = sorted(self.ever_active.symmetric_difference(active_set))
            self.ever_active = self.ever_active.union(active_set)

            if diff:
                # carry out a test
                ever_active = sorted(self.ever_active)
                overall_upper_bd = np.ones(p) * np.inf
                overall_lower_bd = -np.ones(p) * np.inf
                
                always_inactive = np.ones(p, np.bool)
                always_inactive[ever_active] = 0
                
                overall_upper_bd = overall_upper_bd[always_inactive]
                overall_lower_bd = overall_lower_bd[always_inactive]

                for j in range(idx):

                    cur_fit = np.dot(self.X, self.soln_path[j])
                    fit_inactive = np.dot(self.X[:,always_inactive].T, cur_fit)

                    upper_bd = (self.lambda_values[j] * 
                                self.penalty.weights[always_inactive]
                                + fit_inactive)

                    lower_bd = (-self.lambda_values[j] * 
                                self.penalty.weights[always_inactive]
                                + fit_inactive)

                    overall_upper_bd = np.minimum(upper_bd, overall_upper_bd)
                    overall_lower_bd = np.minimum(lower_bd, overall_lower_bd)

                if (np.any(np.dot(self.X[:,always_inactive].T, self.y) > overall_upper_bd) or np.any(np.dot(self.X[:,always_inactive].T, self.y) < overall_lower_bd)):
                    warnings.warn('bound violator!')

                conU = constraints(self.X[:,always_inactive].T,
                                   overall_upper_bd)
                conL = constraints(-self.X[:,always_inactive].T,
                                   -overall_lower_bd)
                inactive_con = con_stack(conU, conL)
                conditional_con = inactive_con.conditional( \
                    self.X[:,ever_active].T,
                    np.dot(self.X[:,ever_active].T, self.y))

                print conditional_con(self.y)

                pval = gibbs_test(conditional_con,
                                  self.y,
                                  self.X[:,diff[0]],
                                  sigma_known=False,
                                  white=False,
                                  ndraw=ndraw,
                                  burnin=burnin,
                                  UMPU=False,
                                  use_random_directions=False,
                                  tilt=None,
                                  alternative='greater',
                                  test_statistic=test_statistic,
                                  )[0]
                print pval, diff, self.ever_active


def main():

    from selection.algorithms.lasso import instance
    X, y = instance()[:2]
    lam_values = np.linspace(0.05, 1, 20) * np.fabs(np.dot(X.T, y)).max()
    lam_values = lam_values[::-1]

    las_path = lasso_path(X, y, lam_values)
    las_path.fit(None)

if __name__ == "__main__":
    main()
