from __future__ import print_function
import sys
import os
import numpy as np

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
    try:
        sigma_est = sigma_cv_R(r_X, r_y)
    except:
        sigma_est = 1.
    return sigma_est


if __name__ == "__main__":

    ###read input files
    inpath = sys.argv[1]
    outdir = sys.argv[2]

    eGenes = os.path.join(inpath, "eGenes.txt")
    with open(eGenes) as g:
        content = g.readlines()
    content = [x.strip() for x in content]

    for egene in range(len(content)):
        # gene = str(content[egene+1413])
        gene = str(content[egene])

        E = np.load(os.path.join(inpath + "s_" + gene) + ".npy")
        if E.ndim == 0:
            E = np.asarray([E])
        E = (E.reshape((E.shape[0],))).astype(int)

        if E.shape[0] > 0:
            X = np.load(os.path.join(inpath + "X_" + gene) + ".npy")
            n, p = X.shape
            X -= X.mean(0)[None, :]
            X /= (X.std(0)[None, :] * np.sqrt(n))

            prototypes = np.loadtxt(os.path.join(inpath + "protoclust_" + gene) + ".txt", delimiter='\t')
            prototypes = np.unique(prototypes).astype(int)
            X_pruned = X[:, prototypes]

            y = np.load(os.path.join(inpath + "y_" + gene) + ".npy")
            y = y.reshape((y.shape[0],))

            sigma_est = glmnet_sigma(X_pruned, y)
            print("sigma est", sigma_est)
            y /= sigma_est

            E = np.load(os.path.join(inpath + "s_" + gene) + ".npy")
            E = E.reshape((E.shape[0],))
            E = E - 1

            active = np.zeros(p, dtype=bool)
            active[E] = 1
            nactive = active.sum()

            projection_active = X[:, active].dot(np.linalg.inv(X[:, active].T.dot(X[:, active])))

            point_est = projection_active.T.dot(y)
            sd = np.sqrt(np.linalg.inv(X[:, active].T.dot(X[:, active])).diagonal())
            unad_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T
            coverage_unad = np.zeros(nactive)
            unad_length = np.zeros(nactive)

            for k in range(nactive):
                if (unad_intervals[k, 0] <= 0.) and (0. <= unad_intervals[k, 1]):
                    coverage_unad[k] = 1
                unad_length[k] = unad_intervals[k, 1] - unad_intervals[k, 0]

            nsig = 1. - (coverage_unad.sum() / float(nactive))
            output = np.transpose(np.vstack((nactive * np.ones(nactive), unad_intervals[:, 0], unad_intervals[:, 1],
                                             unad_length, point_est, nsig * np.ones(nactive))))

            outfile = os.path.join(outdir + "fwd_bwd_output_" + gene + ".txt")
            np.savetxt(outfile, output)

        else:
            output = np.transpose(np.vstack((0., 0., 0., 0., 0., 0.)))

            outfile = os.path.join(outdir + "fwd_bwd_output_" + gene + ".txt")
            np.savetxt(outfile, output)

        sys.stderr.write("Iteration completed" + str(egene) + "\n")