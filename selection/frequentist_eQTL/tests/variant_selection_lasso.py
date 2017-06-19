from __future__ import print_function
import sys
import os

import numpy as np
import regreg.api as rr
from selection.frequentist_eQTL.estimator import M_estimator_exact

def lasso_selection(X, y, lam_frac=1.):

    from selection.api import randomization

    n, p = X.shape

    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * 1.
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

    lasso_output = np.transpose(np.vstack((active_set, M_est.target_observed)))

    return lasso_output


if __name__ == "__main__":

    path = sys.argv[1]
    outdir = sys.argv[2]
    result = sys.argv[3]

    gene_file = path + "Genes.txt"
    with open(gene_file) as g:
        content = g.readlines()

    content = [x.strip() for x in content]

    intermediate = '/Users/snigdhapanigrahi/simes_output_Liver/egenes/'

    egenes = np.loadtxt(os.path.join(intermediate, "egene_index_" + str(result)) + ".txt")

    if egenes.size ==1:
        egenes = egenes.reshape((1,))

    if egenes[0] >= 0.:
        negenes = egenes.shape[0]

        for j in range(negenes):

            index = int(egenes[j])
            gene_name = content[index]

            outfile = os.path.join(outdir, "lasso_output_" + str(gene_name)) + ".txt"

            X = np.load(os.path.join(path + "X_" + str(gene_name)) + ".npy")
            n = X.shape[0]
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))

            y = np.load(os.path.join(path + "y_" + str(gene_name)) + ".npy")
            y = y.reshape((y.shape[0],))

            lasso_output = lasso_selection(X, y, lam_frac=1.)

            np.savetxt(outfile, lasso_output)








