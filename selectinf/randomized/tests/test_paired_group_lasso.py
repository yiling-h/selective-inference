from __future__ import division, print_function

import numpy as np
import nose.tools as nt
from nose.tools import nottest

import regreg.api as rr

from ..group_lasso import (group_lasso,
                           selected_targets,
                           full_targets,
                           debiased_targets)
from ..paired_group_lasso import paired_group_lasso
from ...tests.instance import gaussian_instance
from ...tests.flags import SET_SEED
from ...tests.decorators import set_sampling_params_iftrue, set_seed_iftrue
from ...algorithms.sqrt_lasso import choose_lambda, solve_sqrt_lasso
from ..randomization import randomization
from ...tests.decorators import rpy_test_safe

def test_paired_group_lasso(n=400,
                           p=100,
                           signal_fac=3,
                           s=5,
                           sigma=3,
                           target='full',
                           rho=0.4,
                           randomizer_scale=.75,
                           ndraw=100000):
    cov = np.array([[2, 1, 1],
                    [1, 2, 1],
                    [1, 1, 1]])
    X = np.random.multivariate_normal(mean=np.zeros((3,)), cov=cov, size=100)

    pgl = paired_group_lasso(X=X, weights=1.0, ridge_term=0.0, randomizer_scale=randomizer_scale)
    pgl.fit()
    print(pgl.beta)

