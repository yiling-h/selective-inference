import regreg.api as rr
import regreg.affine as ra
import numpy as np
from scipy.stats import norm as ndist

def restricted_estimator(loss, active, solve_args={'min_its': 50, 'tol': 1.e-10}):
    """
    Fit a restricted model using only columns `active`.

    Parameters
    ----------

    Mest_loss : objective function
        A GLM loss.

    active : ndarray
        Which columns to use.

    solve_args : dict
        Passed to `solve`.

    Returns
    -------

    soln : ndarray
        Solution to restricted problem.

    """
    X, Y = loss.data

    if not loss._is_transform and hasattr(loss, 'saturated_loss'):  # M_est is a glm
        X_restricted = X[:, active]
        loss_restricted = rr.affine_smooth(loss.saturated_loss, X_restricted)
    else:
        I_restricted = ra.selector(active, ra.astransform(X).input_shape[0], ra.identity((active.sum(),)))
        loss_restricted = rr.affine_smooth(loss, I_restricted.T)
    beta_E = loss_restricted.solve(**solve_args)

    return beta_E


def naive_confidence_intervals(diag_cov, observed, level=0.9):
    """
    Compute naive Gaussian based confidence
    intervals for target.
    Parameters
    ----------

    diag_cov : diagonal of a covariance matrix

    observed : np.float
        A vector of observed data of shape `target.shape`

    alpha : float (optional)
        1 - confidence level.

    Returns
    -------
    intervals : np.float
        Gaussian based confidence intervals.
    """
    alpha = 1 - level
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    quantile = - ndist.ppf(alpha/2)
    LU = np.zeros((2, p))
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        LU[0,j] = observed[j] - sigma * quantile
        LU[1,j] = observed[j] + sigma * quantile
    return LU.T

def naive_pvalues(diag_cov, observed, parameter):
    diag_cov = np.asarray(diag_cov)
    p = diag_cov.shape[0]
    pvalues = np.zeros(p)
    for j in range(p):
        sigma = np.sqrt(diag_cov[j])
        pval = ndist.cdf((observed[j] - parameter[j])/sigma)
        pvalues[j] = 2 * min(pval, 1-pval)
    return pvalues

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

    objective = lambda u: -u.T.dot(conjugate_arg) + u.T.dot(precision).dot(u)/2. \
                          + np.log(1.+ 1./((con_offset - con_linear.dot(u))/ scaling)).sum()
    grad = lambda u: -conjugate_arg + precision.dot(u) - con_linear.T.dot(1./(scaling + con_offset - con_linear.dot(u)) -
                                                                       1./(con_offset - con_linear.dot(u)))
    barrier_hessian = lambda u: con_linear.T.dot(np.diag(-1./((scaling + con_offset-con_linear.dot(u))**2.)
                                                 + 1./((con_offset-con_linear.dot(u))**2.))).dot(con_linear)

    current = feasible_point
    current_value = np.inf

    for itercount in range(nstep):
        cur_grad = grad(current)

        # make sure proposal is feasible

        count = 0
        while True:
            count += 1
            proposal = current - step * cur_grad
            if np.all(con_offset-con_linear.dot(proposal) > 0):
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
