from __future__ import print_function
import sys
import os

from selection.algorithms.lasso import lasso
import numpy as np
from scipy.stats.stats import pearsonr

from rpy2.robjects.packages import importr
from rpy2 import robjects

glmnet = importr('glmnet')
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()

def glmnet_sigma(X, y):
    robjects.r('''
                glmnet_cv = function(X,y){
                y = as.matrix(y)
                X = as.matrix(X)

                out = cv.glmnet(X, y, standardize=FALSE, intercept=FALSE)
                lam_minCV = out$lambda.min

                coef = coef(out, s = "lambda.min")
                linear.fit = lm(y~ X[, which(coef>0.001)-1])
                sigma_est = summary(linear.fit)$sigma
                return(sigma_est)
                }''')

    sigma_cv_R = robjects.globalenv['glmnet_cv']
    n, p = X.shape
    r_X = robjects.r.matrix(X, nrow=n, ncol=p)
    r_y = robjects.r.matrix(y, nrow=n, ncol=1)

    sigma_est = sigma_cv_R(r_X, r_y)
    return sigma_est

def lasso_Gaussian(X, y, lam, X_unpruned):

    L = lasso.gaussian(X, y, lam, sigma=1.)

    soln = L.fit()
    active = soln != 0
    print("Lasso estimator", soln[active])
    nactive = active.sum()
    print("nactive", nactive)

    projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))

    active_set = np.nonzero(active)[0]
    print("active set", active_set)

    active_signs = np.sign(soln[active])
    C = L.constraints
    coverage_nonzero = np.zeros(nactive)

    if C is not None:
        one_step = L.onestep_estimator
        print("one step", one_step)
        point_est = projection_active.T.dot(y) 
        sd = np.sqrt(np.linalg.inv(X[:, active].T.dot(X[:, active])).diagonal())
        unad_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T
        unad_length = (unad_intervals[:,1]- unad_intervals[:,0]).sum() / nactive
        unad_norm = np.power(point_est, 2.).sum() /float(nactive)

        for k in range(one_step.shape[0]):
            if (unad_intervals[k, 0] > 0.) or (unad_intervals[k, 1]< 0.):
                coverage_nonzero[k] += 1

        coverage_nonzero = coverage_nonzero.sum() /float(nactive)

        return np.vstack((nactive, coverage_nonzero, unad_length, unad_norm))

    else:
        return np.vstack((0., 0., 0., 0., 0.))

if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]
    count_nosel = 0.

    for egene in range(len(content)):
        gene = str(content[egene])
        if os.path.exists(os.path.join(inpath + "X_" + gene) + ".npy"):
            X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
            n, p = X.shape
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))
            X_unpruned = X

            prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
            prototypes = np.unique(prototypes).astype(int)
            print("prototypes", prototypes.shape[0])
            X = X[:, prototypes]

            y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
            y = y.reshape((y.shape[0],))
  
            try:
                sigma_est = glmnet_sigma(X, y)
            except:
                sigma_est = 1.

            y /= sigma_est

            lam_frac = 1.2
            lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))

            lasso_results = lasso_Gaussian(X,
                                           y,
                                           lam,
                                           X_unpruned)

            if lasso_results[0]== 0.:
                count_nosel +=1

            outfile = os.path.join(outdir + "nonrand_output_" + gene + ".txt")
            np.savetxt(outfile, lasso_results)
            sys.stderr.write("Iteration completed" + str(egene) + "\n")

        else:
            sys.stderr.write("Error" + str(egene) + "\n")

    sys.stderr.write("No of Egenes where Lasso did not select anything" + str(count_nosel) + "\n")