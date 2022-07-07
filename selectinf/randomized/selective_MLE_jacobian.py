from __future__ import division, print_function

import numpy as np, pandas as pd
from scipy.linalg import block_diag
from scipy.stats import norm as ndist
from .selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C
from ..algorithms.barrier_affine import solve_barrier_affine_py
from ..base import target_query_Interactspec

class mle_inference(object):

    def __init__(self,
                 query_spec,
                 target_spec,
                 useJacobian,
                 Jacobian_spec,
                 solve_args={'tol': 1.e-12}):

        self.query_spec = query_spec
        self.target_spec = target_spec
        self.solve_args = solve_args
        self.useJacobian = useJacobian
        self.Jacobian_spec = Jacobian_spec
        
    def solve_estimating_eqn(self,
                             alternatives=None,
                             useC=False,
                             level=0.90):

        # cond_mean, cond_cov, opt_linear, linear_part, offset, M1, M2, M3
        # observed_opt_state : np.ndarray     # gammas
        # observed_score_state : np.ndarray   # -X'Y
        # observed_subgrad : np.ndarray       # subgradients scaled by lambda
        # observed_soln : np.ndarray          # gammas
        # observed_score : np.ndarray         # -X'Y + subgrad = "score_offset"
        QS = self.query_spec

        # observed_target : np.ndarray
        # cov_target : np.ndarray
        # regress_target_score = [(XE'XE)^-1 0_{-E}]
        # alternatives: list
        TS = self.target_spec

        if self.useJacobian:
            # C = V.T.dot(QI).dot(L).dot(V)
            # active_dirs
            JS = self.Jacobian_spec

        # In the context of group Lasso, the quantities U1-U5 are:
        # U1 =
        # U2 = A' {\Omega^-1} A = (regress_score_target.T.dot(self.prec_randomizer).dot(regress_score_target))
        # U3 = \barA' {\bar\Omega^-1} \barA = regress_opt_target.T.dot(prec_opt).dot(regress_opt_target)
        # U4 =
        # U5 = \barA' {\bar\Omega^-1}
        U1, U2, U3, U4, U5= target_query_Interactspec(QS,
                                                      TS.regress_target_score,
                                                      TS.cov_target)

        prec_target = np.linalg.inv(TS.cov_target)

        prec_target_nosel = prec_target + U2 - U3

        # Same as:
        # _P = TS.regress_score_target.T.dot(self.prec_randomizer).dot(resid_score_target)
        _P = -(U1.T.dot(QS.M1.dot(QS.observed_score)) + U2.dot(TS.observed_target))

        # bias_target = -C, where
        # C = cov_target.dot(_P - regress_opt_target.T.dot(prec_opt).dot(resid_mean_opt_target))
        bias_target = TS.cov_target.dot(U1.T.dot(-U4.dot(TS.observed_target)
                                                 + QS.M1.dot(QS.opt_linear.dot(QS.cond_mean))) - _P)
        
        cond_precision = np.linalg.inv(QS.cond_cov)
        conjugate_arg = cond_precision.dot(QS.cond_mean)

        if useC:
            solver = solve_barrier_affine_C
        else:
            solver = solve_barrier_affine_py

        if not self.useJacobian:
            val, soln, hess = solver(conjugate_arg,
                                     cond_precision,
                                     QS.observed_soln,
                                     QS.linear_part,
                                     QS.offset,
                                     **self.solve_args)
        else:
            val, soln, hess = solve_barrier_affine_jacobian_py(conjugate_arg,
                                                               cond_precision,
                                                               QS.observed_soln,
                                                               QS.linear_part,
                                                               QS.offset,
                                                               JS.C,    # for Jacobian
                                                               JS.active_dirs,
                                                               useJacobian=True,
                                                               **self.solve_args)

        # Everything below this line are unchanged and apply to both the Lasso and group Lasso
        final_estimator = TS.cov_target.dot(prec_target_nosel).dot(TS.observed_target) \
                          + TS.regress_target_score.dot(QS.M1.dot(QS.opt_linear)).dot(QS.cond_mean - soln) \
                          - bias_target

        observed_info_natural = prec_target_nosel + U3 - U5.dot(hess.dot(U5.T))

        unbiased_estimator = TS.cov_target.dot(prec_target_nosel).dot(TS.observed_target) - bias_target

        observed_info_mean = TS.cov_target.dot(observed_info_natural.dot(TS.cov_target))

        Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))

        cdf_vals = ndist.cdf(Z_scores)
        pvalues = []

        if alternatives is None:
            alternatives = ['twosided'] * len(cdf_vals)

        for m, _cdf in enumerate(cdf_vals):
            if alternatives[m] == 'twosided':
                pvalues.append(2 * min(_cdf, 1 - _cdf))
            elif alternatives[m] == 'greater':
                pvalues.append(1 - _cdf)
            elif alternatives[m] == 'less':
                pvalues.append(_cdf)
            else:
                raise ValueError('alternative should be in ["twosided", "less", "greater"]')

        alpha = 1. - level

        quantile = ndist.ppf(1 - alpha / 2.)

        intervals = np.vstack([final_estimator - quantile * np.sqrt(np.diag(observed_info_mean)),
                               final_estimator + quantile * np.sqrt(np.diag(observed_info_mean))]).T

        log_ref = val + conjugate_arg.T.dot(QS.cond_cov).dot(conjugate_arg) / 2.

        result = pd.DataFrame({'MLE': final_estimator,
                               'SE': np.sqrt(np.diag(observed_info_mean)),
                               'Zvalue': Z_scores,
                               'pvalue': pvalues,
                               'alternative': alternatives,
                               'lower_confidence': intervals[:, 0],
                               'upper_confidence': intervals[:, 1],
                               'unbiased': unbiased_estimator})

        return result, observed_info_mean, log_ref

def solve_barrier_affine_jacobian_py(conjugate_arg,
                                     precision,
                                     feasible_point,
                                     con_linear,
                                     con_offset,
                                     C,
                                     active_dirs,
                                     useJacobian=True,
                                     step=1,
                                     nstep=2000,
                                     min_its=500,
                                     tol=1.e-12):
    """
    This needs to be updated to actually use the Jacobian information (in self.C)
    arguments
    conjugate_arg: \\bar{\\Sigma}^{-1} \bar{\\mu}
    precision:  \\bar{\\Sigma}^{-1}
    feasible_point: gamma's from fitting
    con_linear: linear part of affine constraint used for barrier function
    con_offset: offset part of affine constraint used for barrier function
    C: V^T Q^{-1} \\Lambda V
    active_dirs:
    """
    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    def objective(gs):
        p1 = -gs.T.dot(conjugate_arg)
        p2 = gs.T.dot(precision).dot(gs) / 2.
        if useJacobian:
            p3 = - jacobian_grad_hess(gs, C, active_dirs)[0]
        else:
            p3 = 0
        p4 = np.log(1. + 1. / ((con_offset - con_linear.dot(gs)) / scaling)).sum()
        return p1 + p2 + p3 + p4

    def grad(gs):
        p1 = -conjugate_arg + precision.dot(gs)
        p2 = -con_linear.T.dot(1. / (scaling + con_offset - con_linear.dot(gs)))
        if useJacobian:
            p3 = - jacobian_grad_hess(gs, C, active_dirs)[1]
        else:
            p3 = 0
        p4 = 1. / (con_offset - con_linear.dot(gs))
        return p1 + p2 + p3 + p4

    def barrier_hessian(gs):  # contribution of barrier and jacobian to hessian
        p1 = con_linear.T.dot(np.diag(-1. / ((scaling + con_offset - con_linear.dot(gs)) ** 2.)
                                      + 1. / ((con_offset - con_linear.dot(gs)) ** 2.))).dot(con_linear)
        if useJacobian:
            p2 = - jacobian_grad_hess(gs, C, active_dirs)[2]
        else:
            p2 = 0
        return p1 + p2

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset - con_linear.dot(proposal) > 0):
                break
            step *= 0.5
            if count >= 40:
                raise ValueError('not finding a feasible point')
        # make sure proposal is a descent

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            proposed_value = objective(proposal)
            if proposed_value <= current_value:
                break
            step *= 0.5
            if count >= 20:
                if not (np.isnan(proposed_value) or np.isnan(current_value)):
                    break
                else:
                    raise ValueError('value is NaN: %f, %f' % (proposed_value, current_value))

        # stop if relative decrease is small

        if np.fabs(current_value - proposed_value) < tol * np.fabs(current_value) and itercount >= min_its:
            current = proposal
            current_value = proposed_value
            break

        current = proposal
        current_value = proposed_value

        if itercount % 4 == 0:
            step *= 2

    hess = np.linalg.inv(precision + barrier_hessian(current))
    return current_value, current, hess

# Jacobian calculations
def calc_GammaMinus(gamma, active_dirs):
    """Calculate Gamma^minus (as a function of gamma vector, active directions)
    """
    to_diag = [[g] * (ug.size - 1) for (g, ug) in zip(gamma, active_dirs.values())]
    return block_diag(*[i for gp in to_diag for i in gp])


def jacobian_grad_hess(gamma, C, active_dirs):
    """ Calculate the log-Jacobian (scalar), gradient (gamma.size vector) and hessian (gamma.size square matrix)
    """
    if C.shape == (0, 0):  # when all groups are size one, C will be an empty array
        return 0, 0, 0
    else:
        GammaMinus = calc_GammaMinus(gamma, active_dirs)

        # eigendecomposition
        #evalues, evectors = eig(GammaMinus + C)

        # log Jacobian
        #J = log(evalues).sum()
        J = np.log(np.linalg.det(GammaMinus + C))

        # inverse
        #GpC_inv = evectors.dot(np.diag(1 / evalues).dot(evectors.T))
        GpC_inv = np.linalg.inv(GammaMinus + C)

        # summing matrix (gamma.size by C.shape[0])
        S = block_diag(*[np.ones((1, ug.size - 1)) for ug in active_dirs.values()])

        # gradient
        grad_J = S.dot(GpC_inv.diagonal())

        # hessian
        hess_J = -S.dot(np.multiply(GpC_inv, GpC_inv.T).dot(S.T))

        return J, grad_J, hess_J
