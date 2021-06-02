import numpy as np
import pandas as pd
from scipy.stats import norm as ndist

def _solve_barrier_affine_py(conjugate_arg,
                             precision,
                             feasible_point,
                             con_linear,
                             con_offset,
                             step=1,
                             nstep=1000,
                             min_its=200,
                             tol=1.e-10):
    scaling = np.sqrt(np.diag(con_linear.dot(precision).dot(con_linear.T)))

    if feasible_point is None:
        feasible_point = 1. / scaling

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u) / 2. \
                          + np.log(1. + 1. / ((con_offset - con_linear.dot(u)) / scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) - con_linear.T.dot(
        1. / (scaling + con_offset - con_linear.dot(u)) -
        1. / (con_offset - con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1. / ((scaling + con_offset - con_linear.dot(u)) ** 2.)
                                                         + 1. / ((con_offset - con_linear.dot(u)) ** 2.))).dot(
        con_linear)

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


def selective_MLE(observed_target,
                  target_cov,
                  target_score_cov,
                  init_soln,  # initial (observed) value of
                  # optimization variables -- used as a
                  # feasible point.  precise value used
                  # only for independent estimator
                  cond_mean,
                  cond_cov,
                  logdens_linear,
                  linear_part,
                  offset,
                  randomizer_prec,
                  score_offset,
                  solve_args={'tol': 1.e-12},
                  level=0.9):
    """
    Selective MLE based on approximation of
    CGF.

    Parameters
    ----------

    observed_target : ndarray
        Observed estimate of target.

    target_cov : ndarray
        Estimated covaraince of target.

    target_score_cov : ndarray
        Estimated covariance of target and score of randomized query.

    init_soln : ndarray
        Feasible point for optimization problem.

    cond_mean : ndarray
        Conditional mean of optimization variables given target.

    cond_cov : ndarray
        Conditional covariance of optimization variables given target.

    logdens_linear : ndarray
        Describes how conditional mean of optimization
        variables varies with target.

    linear_part : ndarray
        Linear part of affine constraints: $\{o:Ao \leq b\}$

    offset : ndarray
        Offset part of affine constraints: $\{o:Ao \leq b\}$

    randomizer_prec : ndarray
        Precision matrix based on randomization covariance

    solve_args : dict, optional
        Arguments passed to solver.

    level : float, optional
        Confidence level.

    useC : bool, optional
        Use python or C solver.

    """

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(target_cov)

    prec_opt = np.linalg.inv(cond_cov)

    target_linear = target_score_cov.T.dot(prec_target)
    target_offset = score_offset - target_linear.dot(observed_target)

    # target_lin determines how the conditional mean of optimization variables
    # vary with target
    # logdens_linear determines how the argument of the optimization density
    # depends on the score, not how the mean depends on score, hence the minus sign

    target_lin = - logdens_linear.dot(target_linear)
    target_off = cond_mean - target_lin.dot(observed_target)

    if np.asarray(randomizer_prec).shape in [(), (0,)]:
        _P = target_linear.T.dot(target_offset) * randomizer_prec
        _prec = prec_target + (target_linear.T.dot(target_linear) * randomizer_prec) - target_lin.T.dot(prec_opt).dot(
            target_lin)
    else:
        _P = target_linear.T.dot(randomizer_prec).dot(target_offset)
        _prec = prec_target + (target_linear.T.dot(randomizer_prec).dot(target_linear)) - target_lin.T.dot(
            prec_opt).dot(target_lin)

    C = target_cov.dot(_P - target_lin.T.dot(prec_opt).dot(target_off))

    conjugate_arg = prec_opt.dot(cond_mean)

    solver = _solve_barrier_affine_py

    val, soln, hess = solver(conjugate_arg,
                             prec_opt,
                             init_soln,
                             linear_part,
                             offset,
                             **solve_args)

    final_estimator = target_cov.dot(_prec).dot(observed_target) \
                      + target_cov.dot(target_lin.T.dot(prec_opt.dot(cond_mean - soln))) + C

    L = target_lin.T.dot(prec_opt)
    observed_info_natural = _prec + L.dot(target_lin) - L.dot(hess.dot(L.T))

    observed_info_mean = target_cov.dot(observed_info_natural.dot(target_cov))

    Z_scores = final_estimator / np.sqrt(np.diag(observed_info_mean))
    pvalues = ndist.cdf(Z_scores)
    pvalues = 2 * np.minimum(pvalues, 1 - pvalues)

    alpha = 1 - level
    quantile = ndist.ppf(1 - alpha / 2.)

    intervals = np.vstack([final_estimator -
                           quantile * np.sqrt(np.diag(observed_info_mean)),
                           final_estimator +
                           quantile * np.sqrt(np.diag(observed_info_mean))]).T

    log_ref = val + conjugate_arg.T.dot(cond_cov).dot(conjugate_arg) / 2.

    result = pd.DataFrame({'MLE': final_estimator,
                           'SE': np.sqrt(np.diag(observed_info_mean)),
                           'Zvalue': Z_scores,
                           'pvalue': pvalues,
                           'lower_confidence': intervals[:, 0],
                           'upper_confidence': intervals[:, 1]})

    return result, observed_info_mean, log_ref