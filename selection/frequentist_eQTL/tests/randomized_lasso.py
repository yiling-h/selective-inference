from __future__ import print_function
import sys
import os
import argparse
import time

import numpy as np
import regreg.api as rr

from selection.frequentist_eQTL.approx_confidence_intervals import approximate_conditional_density
from selection.frequentist_eQTL.estimator import M_estimator_exact

from selection.randomized.query import naive_confidence_intervals
from selection.randomized.query import naive_pvalues

from selection.bayesian.initial_soln import selection, instance
from selection.bayesian.cisEQTLS.Simes_selection import BH_q


def randomized_lasso_selection(X,
                               y,
                               sigma,
                               lam_frac = 1.2,
                               loss='gaussian',
                               randomizer='gaussian'):
    # select active variables using randomized lasso
    from selection.api import randomization

    sys.stderr.write("Running randomized lasso\n")
    start_time = time.time()
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
   
    used_time = time.time() - start_time
    sys.stderr.write("Active set selected by lasso"+str(active_set)+"\n")
    sys.stderr.write("Used: {:.2f}s\n".format(used_time))

    return(M_est)

def randomized_lasso_inference(M_est, n_cores = 1):
    # compute the selective pivot and intervals for the selected set
    active = M_est._overall
    p = len(active)
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    if nactive == 0:
        return None
    sys.stderr.write("Active set selected by lasso"+str(active_set)+"\n")

    # this part is the slowest
    sys.stderr.write("Running inference\n")
    start_time = time.time()
    ci = approximate_conditional_density(M_est, n_cores=n_cores)
    ci.solve_approx()
    used_time = time.time() - start_time
    sys.stderr.write("Used: {:.2f}s\n".format(used_time))

    ci_sel = np.zeros((nactive, 2))
    pivots = np.zeros(nactive)

    for j in xrange(nactive):
        ci_sel[j, :] = np.array(ci.approximate_ci(j))
        pivots[j] = ci.approximate_pvalue(j, 0.)

    out_result = (ci_sel, pivots, active_set, M_est)
    return(out_result)

def evaluate_selection(M_est, beta):
    # evaluate accuracy
    active = M_est._overall
    assert len(active)==len(beta), "mismatch length of true and estimated signals"
    n_disc = np.sum(active)
    true_sig = beta > 0 
    fdp = 1.0 * np.sum(active & ~true_sig) / np.max((n_disc,1))
    power = 1.0 * np.sum(active & true_sig) / np.max((np.sum(true_sig),1))
    result = np.array([n_disc, fdp, power])
    return(result)

def evaluate_inference(out_result, X, beta, bh_level=0.1):
    # evaluate accuracy
    ci_sel, pivots, active_set, M_est = out_result
    n, p = X.shape

    active = M_est._overall
    active_set = np.asarray([i for i in range(p) if active[i]])
    nactive = np.sum(active)

    true_vec = np.linalg.inv(X[:, active].T.dot(X[:, active])).dot(X[:, active].T).dot(X.dot(beta))
    sys.stderr.write("True target to be covered" + str(true_vec) + "\n")

    class target_class(object):
        def __init__(self, target_cov):
            self.target_cov = target_cov
            self.shape = target_cov.shape

    target = target_class(M_est.target_cov)

    ci_naive = naive_confidence_intervals(target, M_est.target_observed)
    naive_pvals = naive_pvalues(target, M_est.target_observed, true_vec)
    naive_covered = np.zeros(nactive, np.bool)
    naive_length = np.zeros(nactive)

    sel_covered = np.zeros(nactive, np.bool)

    sel_length = np.zeros(nactive)
    for j in xrange(nactive):
        sel_length[j] = ci_sel[j, 1] - ci_sel[j, 0]
        if (ci_sel[j, 0] <= true_vec[j]) and (ci_sel[j, 1] >= true_vec[j]):
            sel_covered[j] = 1
        print(ci_sel[j, :])

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
    selection_result = randomized_lasso_selection(X, y, sigma, lam_frac = 1.2)
    print(evaluate_selection(selection_result, beta))
    inference_result = randomized_lasso_inference(selection_result, n_cores=args.n_cores)
    print(evaluate_inference(inference_result, X, beta))
    

def save_data(args):
    print(args)
    n = args.n_obs
    p = args.n_vars 
    s = args.n_sig
    snr = args.snr

    np.random.seed(0) # ensures same X
    sample = instance(n=n, p=p, s=s, sigma=args.sigma, rho=0, snr=snr)

    assert os.path.exists(args.data_dir), "data directory: {} does not exist".format(args.data_dir)

    X = sample.X
    X_fname = os.path.join(args.data_dir,"X.npy")
    np.save(X_fname, X)
    sys.stderr.write("Written data to {}\n".format(X_fname))
     
    for seedn in xrange(args.max_seed):
        np.random.seed(seedn) # ensures different y
        X_d, y, beta, nonzero, sigma = sample.generate_response()
        # assert np.array_equal(X, X_d), "X not correct"
        y_fname = os.path.join(args.data_dir,"y_{}.npy".format(seedn)) 
        np.save(y_fname, y)
        sys.stderr.write("Written data to {}\n".format(y_fname))

        b_fname = os.path.join(args.data_dir,"b_{}.npy".format(seedn)) 
        np.save(b_fname, beta)
        sys.stderr.write("Written data to {}\n".format(b_fname))

def select_only(args):
    print(args)
    # read X data
    X_fname = os.path.join(args.data_dir,"X.npy")
    X = np.load(X_fname)
    for seedn in xrange(args.max_seed):
        # read y data 
        y_fname = os.path.join(args.data_dir,"y_{}.npy".format(seedn)) 
        y = np.load(y_fname)
        # read b data
        b_fname = os.path.join(args.data_dir,"b_{}.npy".format(seedn)) 
        beta = np.load(b_fname)
        # randomize selection
        selection_result = randomized_lasso_selection(X, y, sigma=args.sigma, lam_frac=args.lam)
        # evaluate result
        print(evaluate_selection(selection_result, beta))
        # save active set 
        active = selection_result._overall
        p = len(active)
        active_set = np.asarray([i for i in range(p) if active[i]])
        a_fname = os.path.join(args.out_dir,"a_{}.npy".format(seedn)) 
        np.save(a_fname, np.array(active_set))
        sys.stderr.write("Saving active set to: {}.\n".format(a_fname))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run randomized Lasso and frequentist post-seleciton inference")
    subparsers = parser.add_subparsers()

    # program to test randomized lasso
    command_parser = subparsers.add_parser('test', help='testing parallel processing') 
    command_parser.add_argument('-n','--n_cores', type=int, default=1, help="number of processers to use")
    command_parser.set_defaults(func=do_test)

    command_parser = subparsers.add_parser('gendata', help='generate data and save to file') 
    command_parser.add_argument('-s','--max_seed', type=int, default=50, help="maximum numbers of seeds to use")
    command_parser.add_argument('-o','--data_dir', default='/scratch/users/jjzhu/sim_eqtl_data', help="directory to save simulated data")
    command_parser.add_argument('-r','--snr', type=float, default=5.0, help="signal to noise ratio")
    command_parser.add_argument('-S','--sigma', type=float, default=1.0, help="variance of noise")
    command_parser.add_argument('-m','--n_sig', type=int, default=0, help="number of signals")
    command_parser.add_argument('-p','--n_vars', type=int, default=7000, help="number of variables")
    command_parser.add_argument('-n','--n_obs', type=int, default=350, help="number of observations")
    command_parser.set_defaults(func=save_data)

    command_parser = subparsers.add_parser('onlyselect', help='only apply randomized lasso selection') 
    command_parser.add_argument('-s','--max_seed', type=int, default=50, help="maximum numbers of seeds to use")
    command_parser.add_argument('-d','--data_dir', default='/scratch/users/jjzhu/sim_eqtl_data', help="directory to read simulated data")
    command_parser.add_argument('-o','--out_dir', default='/scratch/users/jjzhu/sim_eqtl_data', help="directory to save results")
    command_parser.add_argument('-l','--lam', type=float, default=1.2, help="directory to save results")
    command_parser.add_argument('-S','--sigma', type=float, default=1.0, help="variance of noise")
    command_parser.set_defaults(func=select_only)
    ARGS = parser.parse_args()
    if ARGS.func is None:
        parser.print_help()
        sys.exit(1)
    else:
        ARGS.func(ARGS)
