import numpy as np
import pandas as pd
import nose.tools as nt
import seaborn as sns
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import multiprocessing
# from multiprocess import Pool

import regreg.api as rr

from selectinf.randomized.lasso import (lasso, split_lasso)
from selectinf.randomized.group_lasso_query import (group_lasso,
                                 split_group_lasso)

from selectinf.base import (full_targets,
                     selected_targets,
                     debiased_targets)
from selectinf.randomized.tests.instance import (gaussian_group_instance,
                                                 logistic_group_instance)

from selectinf.base import restricted_estimator
import scipy.stats

from selectinf.randomized.tests.test_group_lasso_simulation import (calculate_F1_score,
                                                                  naive_inference,
                                                                  randomization_inference,
                                                                  randomization_inference_fast,
                                                                  split_inference,
                                                                    data_splitting,
                                                                    posterior_inference)


if __name__ == '__main__':
    comparison_logistic_lasso_vary_s()


