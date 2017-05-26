from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
from selection.bayesian.cisEQTLS.Simes_selection import BH_q

pivots = np.array([1.94209967e-07,9.11730591e-01,3.95903637e-01,9.99165822e-01,
                   5.94777056e-01,4.69779629e-01,5.82127910e-01,9.97245916e-02])
p_BH = BH_q(pivots, 0.20)