import numpy as np
from selection.randomized.query import naive_confidence_intervals
from selection.tests.instance import gaussian_group_instance
from selection.randomized.group_lasso import group_lasso, posterior


def coverage_experiment(traj):
    np.random.seed(seed=traj.seed)

    X, Y, beta = draw_data(traj)

    # below calls will update traj with results
    posi(traj, X, Y, beta)
    naive_inference(traj, X, Y, beta)
    data_splitting(traj, X, Y, beta)


def draw_data(traj):
    np.random.seed(seed=traj.seed)

    inst = gaussian_group_instance

    if type(traj.signal_fac) is tuple:
        signal_l = np.sqrt(traj.signal_fac[0] * 2 * np.log(traj.p))
        signal_u = np.sqrt(traj.signal_fac[1] * 2 * np.log(traj.p))
        signal = (signal_l, signal_u)
    else:
        signal = np.sqrt(traj.signal_fac * 2 * np.log(traj.p))

    X, Y, beta = inst(n=traj.n,
                      p=traj.p,
                      signal=signal,
                      groups=traj.groups,
                      sgroup=traj.sgroup,
                      sigma=traj.sigma,
                      rho=traj.rho,
                      equicorrelated=False,
                      random_signs=True)[:3]

    traj.f_add_result('data.X', X)
    traj.f_add_result('data.Y', Y)
    traj.f_add_result('data.beta', beta)

    return X, Y, beta


def grp_lasso_selection(X, Y, traj, randomize=True):
    sigma_ = np.std(Y)          # sigma-hat
    if traj.n > traj.p:
        dispersion = np.linalg.norm(Y - X.dot(np.linalg.pinv(X).dot(Y))) ** 2 / (traj.n - traj.p)
    else:
        dispersion = sigma_ ** 2

    sigma_ = np.sqrt(dispersion)

    grps_gsizes = zip(*np.unique(traj.groups, return_counts=True))  # useful iterable

    weights = dict([(i, traj.weight_frac * sigma_ * np.sqrt(2 * np.log(traj.p)) * np.sqrt(gsize)) for (i, gsize) in grps_gsizes])

    if traj.std:                # standardized mode
        W = np.zeros_like(X)
        for grp in np.unique(traj.groups):
            svdg = np.linalg.svd(X[:, traj.groups == grp],
                                 full_matrices=False, compute_uv=True)
            Wg = svdg[0]
            W[:, traj.groups == grp] = Wg
        X = W                   # overwrite X with standardized W

    if randomize:
        randomizer_scale = traj.randomizer_scale * sigma_
        conv = group_lasso.gaussian(X,
                                    Y,
                                    traj.groups,
                                    weights,
                                    randomizer_scale=randomizer_scale)
    else:
        perturb = np.repeat(0, traj.p)
        conv = group_lasso.gaussian(X,
                                    Y,
                                    traj.groups,
                                    weights,
                                    perturb=perturb)

    signs, _ = conv.fit()
    nonzero = signs != 0

    return nonzero, conv, dispersion


def naive_inference(traj, X, Y, beta):

    nonzero, conv, _ = grp_lasso_selection(X, Y, traj, randomize=False)

    if nonzero.sum() > 0:
        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        fe, oi, z, p, intervals, _ = conv.naive_inference()

        lci = intervals[:, 0]
        uci = intervals[:, 1]

        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result('naive.componentwise.coverage', coverage)
        traj.f_add_result('naive.componentwise.length', length)

        traj.f_add_result('naive.mean.coverage', np.mean(coverage))
        traj.f_add_result('naive.mean.length', np.mean(length))
    else:
        traj.f_add_result('naive.componentwise.coverage', np.nan)
        traj.f_add_result('naive.componentwise.length', np.nan)

        traj.f_add_result('naive.mean.coverage', np.nan)
        traj.f_add_result('naive.mean.length', np.nan)


def posi(traj, X, Y, beta):
    nonzero, conv, dispersion = grp_lasso_selection(X, Y, traj, randomize=True)

    if nonzero.sum() > 0:
        conv._setup_implied_gaussian()

        def prior(target_parameter, prior_var=100 * dispersion):
            grad_prior = -target_parameter / prior_var
            log_prior = -np.linalg.norm(target_parameter) ** 2 / (2. * prior_var)
            return grad_prior, log_prior


        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        if len(np.unique(traj.groups)) == traj.p:  # if groups are all atomic
            print('I am atomic!')
            useJacobian = False
        else:
            useJacobian = True

        if traj.std:
            posterior_inf = posterior(conv,  #  this sometimes takes a long time to run
                                      prior=prior,
                                      dispersion=dispersion,
                                      useJacobian=useJacobian,
                                      XrawE=X[:, nonzero])
        else:
            posterior_inf = posterior(conv,  #  this sometimes takes a long time to run
                                      prior=prior,
                                      dispersion=dispersion,
                                      useJacobian=useJacobian)

        samples = posterior_inf.langevin_sampler(nsample=1500,
                                                 nburnin=100,
                                                 step=1.,
                                                 verbose=0)

        traj.f_add_result('samples', samples)

        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result('posi.componentwise.coverage', coverage)
        traj.f_add_result('posi.componentwise.length', length)

        traj.f_add_result('posi.mean.coverage', np.mean(coverage))
        traj.f_add_result('posi.mean.length', np.mean(length))
    else:
        traj.f_add_result('posi.componentwise.coverage', np.nan)
        traj.f_add_result('posi.componentwise.length', np.nan)

        traj.f_add_result('posi.mean.coverage', np.nan)
        traj.f_add_result('posi.mean.length', np.nan)


def data_splitting(traj, X, Y, beta):
    n = X.shape[0]
    n_train = int(np.floor(n/2))

    indices = np.random.permutation(n)

    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train = X[train_idx, :]
    Y_train = Y[train_idx]

    X_test = X[test_idx, :]
    Y_test = Y[test_idx]

    nonzero, conv, _ = grp_lasso_selection(X_train, Y_train, traj, randomize=False)

    if nonzero.sum() > 0:
        beta_target = np.linalg.pinv(X_test[:, nonzero]).dot(X_test.dot(beta))

        QI = np.linalg.inv(X_test[:, nonzero].T.dot(X_test[:, nonzero]))

        observed_target = np.linalg.pinv(X_test[:, nonzero]).dot(Y_test)

        dispersion = np.sum((Y_test - X_test[:, nonzero].dot(observed_target)) ** 2) / (X_test[:, nonzero].shape[0] - X_test[:, nonzero].shape[1])

        cov_target = QI * dispersion

        intervals = naive_confidence_intervals(np.diag(cov_target), observed_target)

        lci = intervals[:, 0]
        uci = intervals[:, 1]

        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result('split.componentwise.coverage', coverage)
        traj.f_add_result('split.componentwise.length', length)

        traj.f_add_result('split.mean.coverage', np.mean(coverage))
        traj.f_add_result('split.mean.length', np.mean(length))
    else:
        traj.f_add_result('split.componentwise.coverage', np.nan)
        traj.f_add_result('split.componentwise.length', np.nan)

        traj.f_add_result('split.mean.coverage', np.nan)
        traj.f_add_result('split.mean.length', np.nan)
