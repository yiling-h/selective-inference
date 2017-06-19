from __future__ import print_function
import sys
import os
from scipy.stats import norm as normal

import numpy as np
import regreg.api as rr
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.bayesian.selection_probability_rr import nonnegative_softmax_scaled
from selection.frequentist_eQTL.approx_confidence_intervals import neg_log_cube_probability

from selection.randomized.M_estimator import M_estimator
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues
from selection.tests.instance import gaussian_instance
from selection.api import randomization


path = '/Users/snigdhapanigrahi/Liver_lasso/liver/'
gene_file = path + "Genes.txt"
with open(gene_file) as g:
    content = g.readlines()

content = [x.strip() for x in content]
print("desired content id", content[86])

X = np.load(os.path.join(path + "X_" + str(content[86])) + ".npy")
n, p = X.shape
print(p)
X -= X.mean(0)[None, :]
X /= (X.std(0)[None, :] * np.sqrt(n))

y = np.load(os.path.join(path + "y_" + str(content[86])) + ".npy")
y = y.reshape((y.shape[0],))


lam = 1. * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * 1.
loss = rr.glm.gaussian(X, y)

epsilon = 1. / np.sqrt(n)

W = np.ones(p) * lam
penalty = rr.group_lasso(np.arange(p),
                         weights=dict(zip(np.arange(p), W)), lagrange=1.)

randomization = randomization.isotropic_gaussian((p,), scale=1.)

M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer='gaussian')

M_est.solve_approx()
active = M_est._overall
active_set = np.asarray([i for i in range(p) if active[i]])
nactive = np.sum(active)
sys.stderr.write("Active set selected by lasso" + str(active_set) + "\n")
sys.stderr.write("Observed target" + str(M_est.target_observed)+ "\n")



