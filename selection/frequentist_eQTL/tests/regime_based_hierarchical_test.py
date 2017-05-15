from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.estimator import M_estimator_2step
from selection.frequentist_eQTL.approx_ci_2stage import approximate_conditional_prob_2stage, approximate_conditional_density_2stage

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

#from selection.bayesian.initial_soln import instance
from selection.frequentist_eQTL.simes_BH_selection import BH_selection_egenes, simes_selection_egenes
from selection.bayesian.cisEQTLS.Simes_selection import BH_q
from selection.frequentist_eQTL.instance import instance


