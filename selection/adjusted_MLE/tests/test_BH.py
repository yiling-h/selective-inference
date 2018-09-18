import numpy as np
from scipy.stats import norm as ndist

import rpy2.robjects as rpy
from rpy2.robjects import numpy2ri

from selection.tests.instance import gaussian_instance
from selection.tests.decorators import rpy_test_safe

from selection.randomized.screening import stepup, stepup_selection
from selection.randomized.randomization import randomization

def BHfilter(pval, q=0.2):
    numpy2ri.activate()
    rpy.r.assign('pval', pval)
    rpy.r.assign('q', q)
    rpy.r('Pval = p.adjust(pval, method="BH")')
    rpy.r('S = which((Pval < q)) - 1')
    S = rpy.r('S')
    numpy2ri.deactivate()
    return np.asarray(S, np.int)

def test_BH(p=500,
            s=50,
            sigma=3.,
            rho=0.35,
            randomizer_scale=1.,
            use_MLE=True,
            marginal=False,
            level=0.9):

