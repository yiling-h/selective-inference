from __future__ import print_function
import sys
import os
import argparse

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_density
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.cisEQTLS.Simes_selection import BH_q


def randomized_lasso_trial(X,
                           y,
                           beta,
                           sigma,
                           bh_level,
                           lam_frac = 1.2,
                           loss='gaussian',
                           randomizer='gaussian',
                           n_cores = 1):

    from selection.api import randomization
    if beta[0] == 0:
        s = 0

    n, p = X.shape
    if loss == "gaussian":
        lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0)) * sigma
        loss = rr.glm.gaussian(X, y)

    epsilon = 1. / np.sqrt(n)

    W = np.ones(p) * lam
    penalty = rr.group_lasso(np.arange(p),
                             weights=dict(zip(np.arange(p), W)), lagrange=1.)
    if randomizer == 'gaussian':
        randomization = randomization.isotropic_gaussian((p,), scale=1.)
    elif randomizer == 'laplace':
        randomization = randomization.laplace((p,), scale=1.)

    M_est = M_estimator_exact(loss, epsilon, penalty, randomization, randomizer)
    M_est.solve_approx()
    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    print("lasso selects", nactive)

    if nactive == 0:
        return None
    sys.stderr.write("Active set selected by lasso"+str(active_set)+"\n")

    true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
    sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

    # save M_est to file
    

    # this part is the slowest
    ci = approximate_conditional_density(M_est, n_cores=n_cores)
    ci.solve_approx()



    ci_sel = np.zeros((nactive, 2))
    sel_covered = np.zeros(nactive, np.bool)
    sel_length = np.zeros(nactive)
    pivots = np.zeros(nactive)

    class target_class(object):
        def __init__(self, target_cov):
            self.target_cov = target_cov
            self.shape = target_cov.shape

    target = target_class(M_est.target_cov)

    ci_naive = naive_confidence_intervals(target, M_est.target_observed)
    naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)
    naive_covered = np.zeros(nactive, np.bool)
    naive_length = np.zeros(nactive)

    for j in xrange(nactive):
        ci_sel[j, :] = np.array(ci.approximate_ci(j))
        if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
            sel_covered[j] = 1
        sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
        print(ci_sel[j, :])
        pivots[j] = ci.approximate_pvalue(j, 0.)

        # naive ci
        if (ci_naive[j, 0] <= true_vec[j]) and (ci_naive[j, 1] >= true_vec[j]):
            naive_covered[j] += 1
        naive_length[j] = ci_naive[j, 1] - ci_naive[j, 0]

    p_BH = BH_q(pivots, bh_level)
    discoveries_active = np.zeros(nactive)
    if p_BH is not None:
        for indx in p_BH[1]:
            discoveries_active[indx] = 1

    list_results = np.transpose(np.vstack((sel_covered,
                                           sel_length,
                                           pivots,
                                           naive_covered,
                                           naive_pvals,
                                           naive_length,
                                           active_set,
                                           discoveries_active)))


    print("list of results", list_results)
    return list_results


def do_test(args):
    seedn = 1
    n = 10
    p = 10
    s = 1
    snr = 5.
    bh_level = 0.10
    np.random.seed(9999) # ensures same X
    sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)
    np.random.seed(seedn) # ensures different y
    X, y, beta, nonzero, sigma = sample.generate_response()
    # random_lasso = randomized_lasso_trial(X, y, beta, sigma, bh_level, n_cores=args.n_cores)


def save_data(args):

    n = 350
    p = 7000
    s = 3
    snr = 5.
    bh_level = 0.10
    np.random.seed(0) # ensures same X
    sample = instance(n=n, p=p, s=s, sigma=1., rho=0, snr=snr)

    assert os.path.exists(args.data_dir), "data directory: {} does not exist".format(args.data_dir)

    X = sample.X
    X_fname = os.path.join(args.data_dir,"X.npy")
    np.save(X_fname, X)
    sys.stderr.write("Written data to {}\n".format(X_fname))
     
    for seedn in xrange(args.max_seed):
        np.random.seed(seedn) # ensures different y
        X_d, y, beta, nonzero, sigma = sample.generate_response()
        assert np.array_equal(X, X_d), "X not correct"
        y_fname = os.path.join(args.data_dir,"y_{}.npy".format(seedn)) 
        np.save(y_fname, y)
        sys.stderr.write("Written data to {}\n".format(y_fname))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run randomized Lasso and frequentist post-seleciton inference")
    subparsers = parser.add_subparsers()

    # program to test randomized lasso
    command_parser = subparsers.add_parser('test', help='testing parallel processing') 
    command_parser.add_argument('-n','--n_cores', type=int, default=1, help="number of processers to use")
    command_parser.set_defaults(func=do_test)

    command_parser = subparsers.add_parser('gendata', help='generate data and save to file') 
    command_parser.add_argument('-s','--max_seed', type=int, default=49, help="maximum numbers of seeds to use")
    command_parser.add_argument('-o','--data_dir', default='/scratch/users/jjzhu/sim_eqtl_data', help="directory to save simulated data")
    command_parser.set_defaults(func=save_data)

    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
