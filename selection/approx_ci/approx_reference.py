from __future__ import division, print_function

import numpy as np
import nose.tools as nt, functools

import regreg.api as rr

from selection.randomized.lasso import lasso, carved_lasso, selected_targets, full_targets, debiased_targets
from selection.tests.instance import gaussian_instance, nonnormal_instance, mixed_normal_instance
from selection.tests.flags import SET_SEED
from selection.tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from selection.algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from selection.randomized.randomization import randomization
from selection.tests.decorators import rpy_test_safe

from rpy2 import robjects
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scipy.stats as stats
from selection.randomized.selective_MLE_utils import solve_barrier_affine as solve_barrier_affine_C

def approx_reference(grid,
                     observed_target,
                     cov_target,
                     cov_target_score,
                     init_soln,
                     cond_mean,
                     cond_cov,
                     logdens_linear,
                     linear_part,
                     offset,
                     solve_args={'tol': 1.e-15}
                     ):

    if np.asarray(observed_target).shape in [(), (0,)]:
        raise ValueError('no target specified')

    observed_target = np.atleast_1d(observed_target)
    prec_target = np.linalg.inv(cov_target)
    target_lin = - logdens_linear.dot(cov_target_score.T.dot(prec_target))

    prec_opt = np.linalg.inv(cond_cov)

    ref_hat =[]
    solver = solve_barrier_affine_C
    for k in range(grid.shape[0]):
        cond_mean_grid = target_lin.dot(np.asarray([grid[k]])) + (cond_mean - target_lin.dot(observed_target))
        conjugate_arg = prec_opt.dot(cond_mean_grid)

        val, _, _ = solver(conjugate_arg,
                           prec_opt,
                           init_soln,
                           linear_part,
                           offset,
                           **solve_args)

        ref_hat.append(-val-(conjugate_arg.T.dot(cond_cov).dot(conjugate_arg)/2.))

    return np.asarray(ref_hat)

def approx_density(grid,
                   mean_parameter,
                   cov_target,
                   approx_log_ref):

    _approx_density = []
    for k in range(grid.shape[0]):
        _approx_density.append(np.exp(-np.true_divide((grid[k] - mean_parameter) ** 2, 2 * cov_target)+ approx_log_ref[k]))
    _approx_density_ = np.asarray(_approx_density)/(np.asarray(_approx_density).sum())
    return np.cumsum(_approx_density_)

def approx_ci(param_grid,
              grid,
              cov_target,
              approx_log_ref,
              indx_obsv):

    area = np.zeros(param_grid.shape[0])

    for k in range(param_grid.shape[0]):
        area_vec = approx_density(grid,
                                  param_grid[k],
                                  cov_target,
                                  approx_log_ref)
        area[k] = area_vec[indx_obsv]

    region = param_grid[(area >= 0.05) & (area <= 0.95)]
    if region.size > 0:
        return np.nanmin(region), np.nanmax(region)
    else:
        return 0., 0.




