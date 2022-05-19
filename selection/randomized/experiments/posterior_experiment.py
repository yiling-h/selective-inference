import numpy as np
from selection.randomized.query import naive_confidence_intervals
from selection.randomized.tests.instance import gaussian_group_instance
from selection.randomized.group_lasso import group_lasso, posterior
from time import perf_counter as clk


def coverage_experiment(traj):
    cseed = traj.seed + traj.v_idx  # seed offset + run idx
    traj.f_add_result('cseed', cseed)
    np.random.seed(seed=cseed)

    X, Y, beta = draw_data(traj)

    # below calls will update traj with results
    naive_inference(traj, X, Y, beta)
    data_splitting(traj, X, Y, beta, splitrat=0.5)
    data_splitting(traj, X, Y, beta, splitrat=0.67)
    data_splitting(traj, X, Y, beta, splitrat=0.33)
    posi(traj, X, Y, beta, splitrat=0.5)
    posi(traj, X, Y, beta, splitrat=0.67)
    posi(traj, X, Y, beta, splitrat=0.33)


def draw_data(traj):
    inst = gaussian_group_instance

    if type(traj.signal_fac) is tuple:
        signal_l = np.sqrt(traj.signal_fac[0] * 2 * np.log(traj.p))
        signal_u = np.sqrt(traj.signal_fac[1] * 2 * np.log(traj.p))
        signal = (signal_l, signal_u)
    else:
        signal = np.sqrt(traj.signal_fac * 2 * np.log(traj.p))

    sgroup = traj.sgroup        # pass thru defaults
    groups = traj.groups        # pass thru defaults

    if traj.og:  # OG mode
        groups = {}
        for k in range(0, 34):
            groups[k] = range(k*3, k*3 + 4)

        sgroup = [0]     # only the 0th group is active
        active_feats = np.array([], dtype=np.int)
        for g in range(traj.sgroup):
            active_feats = np.union1d(active_feats, groups[g])
        groups = np.ones(traj.p, dtype=np.int)
        groups[active_feats] = 0

    X, Y, beta = inst(n=traj.n,
                      p=traj.p,
                      signal=signal,
                      groups=groups,
                      sgroup=sgroup,
                      sigma=traj.sigma,
                      rho=traj.rho,
                      equicorrelated=False,
                      random_signs=True)[:3]

    traj.f_add_result('data.X', X)
    traj.f_add_result('data.Y', Y)
    traj.f_add_result('data.beta', beta)

    return X, Y, beta


def grp_lasso_selection(X, Y, traj, randomize=True, eta=0):
    dispersion = traj.sigma ** 2  # use the true (oracle) value of sigma^2 as dispersion
    sigma_ = np.sqrt(dispersion)

    W = X.copy()                # use X as W if we don't change it

    if traj.og:
        print("Running in OG mode")
        groups = {}
        for k in range(0, 34):
            groups[k] = range(k*3, k*3 + 4)

        W = np.hstack([X[:, inds] for inds in groups.values()])
        grps = np.arange(len(groups)).repeat([len(g) for g in groups.values()])
        mean_diag = np.mean((W ** 2).sum(0))
        ridge_term = np.sqrt(mean_diag)/np.sqrt(traj.n-1)
        print(ridge_term)
    else:
        ridge_term = 0.
        grps = traj.groups

    grps_gsizes = zip(*np.unique(grps, return_counts=True))  # useful iterable

    avg_gsize_flr = np.floor(np.mean(np.unique(grps, return_counts=True)[1]))

    n_scale = W.shape[0] / traj.n
    if traj.std:                # scale differently for standardized
        n_scale = np.sqrt(n_scale)

    weights = dict([(i, n_scale * traj.weight_frac * sigma_ * np.sqrt(2 * np.log(traj.p)) * np.sqrt(gsize) / np.sqrt(avg_gsize_flr)) for (i, gsize) in grps_gsizes])

    if traj.std:                # standardized mode
        print("Running in standardized mode")
        W = np.zeros_like(X)
        for grp in np.unique(grps):
            svdg = np.linalg.svd(X[:, grps == grp],
                                 full_matrices=False, compute_uv=True)
            Wg = svdg[0]
            W[:, grps == grp] = Wg

    if randomize:
        conv = group_lasso.gaussian(W,
                                    Y,
                                    grps,
                                    weights,
                                    randomizer_scale=eta,
                                    ridge_term=ridge_term)
    else:
        perturb = np.repeat(0, W.shape[1])
        conv = group_lasso.gaussian(W,
                                    Y,
                                    grps,
                                    weights,
                                    perturb=perturb,
                                    ridge_term=ridge_term)

    signs, _ = conv.fit()
    nonzero = signs != 0
    print(f'I selected {np.sum(nonzero)} coefficients as nonzero.\n')

    if traj.og:                 # map nonzero back to original indexing
        groups = {}
        for k in range(0, 34):
            groups[k] = range(k*3, k*3 + 4)
        back_grp_map = np.hstack([inds for inds in groups.values()])
        nonzero_raw_inds = back_grp_map[nonzero]
        nonzero_raw = np.repeat([False], X.shape[1])
        for ind in nonzero_raw_inds:
            nonzero_raw[ind] = True
        nonzero = nonzero_raw

    return nonzero, conv, dispersion


def naive_inference(traj, X, Y, beta):
    tic = clk()
    nonzero, conv, dispersion = grp_lasso_selection(X, Y, traj, randomize=False)
    traj.f_add_result('naive.nonzero.mask', nonzero)
    traj.f_add_result('naive.nonzero.nnz', nonzero.sum())

    nz_true = beta.astype(bool)
    traj.f_add_result('naive.sigdet.tp', np.logical_and(nonzero, nz_true).sum())
    traj.f_add_result('naive.sigdet.tn', np.logical_and(~nonzero, ~nz_true).sum())
    traj.f_add_result('naive.sigdet.fp', np.logical_and(nonzero, ~nz_true).sum())
    traj.f_add_result('naive.sigdet.fn', np.logical_and(~nonzero, nz_true).sum())

    if nonzero.sum() > 0:
        prior_var = 100 * dispersion

        mupos, sigmapos = compute_posterior_distribution(prior_var,
                                                         dispersion,
                                                         X[:, nonzero],
                                                         Y)

        beta_target = np.linalg.pinv(X[:, nonzero]).dot(X.dot(beta))

        traj.f_add_result('naive.mse.target', compute_mse_target(mupos, beta_target))
        traj.f_add_result('naive.mse.truth', compute_mse_truth(mupos, beta, nonzero))

        intervals = naive_confidence_intervals(np.diag(sigmapos), mupos)

        lci = intervals[:, 0]
        uci = intervals[:, 1]

        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result('naive.componentwise.coverage', coverage)
        traj.f_add_result('naive.componentwise.length', length)

        traj.f_add_result('naive.mean.coverage', np.mean(coverage))
        traj.f_add_result('naive.mean.length', np.mean(length))

        post_coverage = compute_coverage(lci, uci, beta, nonzero)
        traj.f_add_result('naive.postcoverage', np.mean(post_coverage))

        tp, tn, fp, fn = compute_post_sigdet(lci, uci, beta, nonzero)
        traj.f_add_result('naive.postsigdet.tp', tp)
        traj.f_add_result('naive.postsigdet.tn', tn)
        traj.f_add_result('naive.postsigdet.fp', fp)
        traj.f_add_result('naive.postsigdet.fn', fn)
    else:
        traj.f_add_result('naive.mse.target', np.nan)
        traj.f_add_result('naive.mse.truth', compute_mse_truth(0, beta, nonzero))

        traj.f_add_result('naive.componentwise.coverage', np.nan)
        traj.f_add_result('naive.componentwise.length', np.nan)

        traj.f_add_result('naive.mean.coverage', np.nan)
        traj.f_add_result('naive.mean.length', np.nan)

        traj.f_add_result('naive.postcoverage', 1 - np.mean(nz_true))

        traj.f_add_result('naive.postsigdet.tp', 0)
        traj.f_add_result('naive.postsigdet.tn', np.sum(~nz_true))
        traj.f_add_result('naive.postsigdet.fp', 0)
        traj.f_add_result('naive.postsigdet.fn', np.sum(nz_true))

    toc = clk()
    traj.f_add_result('naive.runtime', toc - tic)

def posi(traj, X, Y, beta, splitrat):
    tic = clk()
    eta = traj.sigma * np.sqrt((1 - splitrat)/splitrat)
    nonzero, conv, dispersion = grp_lasso_selection(X, Y, traj, randomize=True, eta=eta)
    traj.f_add_result(f'posi{splitrat}.nonzero.mask', nonzero)
    traj.f_add_result(f'posi{splitrat}.nonzero.nnz', nonzero.sum())

    nz_true = beta.astype(bool)
    traj.f_add_result(f'posi{splitrat}.sigdet.tp', np.logical_and(nonzero, nz_true).sum())
    traj.f_add_result(f'posi{splitrat}.sigdet.tn', np.logical_and(~nonzero, ~nz_true).sum())
    traj.f_add_result(f'posi{splitrat}.sigdet.fp', np.logical_and(nonzero, ~nz_true).sum())
    traj.f_add_result(f'posi{splitrat}.sigdet.fn', np.logical_and(~nonzero, nz_true).sum())

    if (nonzero.sum() > 0) and (traj.nsample > 0):
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

        if traj.std or traj.og:
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


        samples = posterior_inf.langevin_sampler(nsample=traj.nsample,
                                                 nburnin=100,
                                                 step=1.,
                                                 verbose=0)

        mupos = np.mean(samples, 0)

        traj.f_add_result(f'posi{splitrat}.mse.target', compute_mse_target(mupos, beta_target))
        traj.f_add_result(f'posi{splitrat}.mse.truth', compute_mse_truth(mupos, beta, nonzero))

        traj.f_add_result(f'posi{splitrat}.samples', samples)

        lci = np.percentile(samples, 5, axis=0)
        uci = np.percentile(samples, 95, axis=0)
        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result(f'posi{splitrat}.componentwise.coverage', coverage)
        traj.f_add_result(f'posi{splitrat}.componentwise.length', length)

        traj.f_add_result(f'posi{splitrat}.mean.coverage', np.mean(coverage))
        traj.f_add_result(f'posi{splitrat}.mean.length', np.mean(length))

        post_coverage = compute_coverage(lci, uci, beta, nonzero)
        traj.f_add_result(f'posi{splitrat}.postcoverage', np.mean(post_coverage))

        tp, tn, fp, fn = compute_post_sigdet(lci, uci, beta, nonzero)
        traj.f_add_result(f'posi{splitrat}.postsigdet.tp', tp)
        traj.f_add_result(f'posi{splitrat}.postsigdet.tn', tn)
        traj.f_add_result(f'posi{splitrat}.postsigdet.fp', fp)
        traj.f_add_result(f'posi{splitrat}.postsigdet.fn', fn)
    else:
        traj.f_add_result(f'posi{splitrat}.mse.target', np.nan)
        traj.f_add_result(f'posi{splitrat}.mse.truth', compute_mse_truth(0, beta, nonzero))

        traj.f_add_result(f'posi{splitrat}.componentwise.coverage', np.nan)
        traj.f_add_result(f'posi{splitrat}.componentwise.length', np.nan)

        traj.f_add_result(f'posi{splitrat}.mean.coverage', np.nan)
        traj.f_add_result(f'posi{splitrat}.mean.length', np.nan)

        traj.f_add_result(f'posi{splitrat}.postcoverage', 1 - np.mean(nz_true))


        traj.f_add_result(f'posi{splitrat}.postsigdet.tp', 0)
        traj.f_add_result(f'posi{splitrat}.postsigdet.tn', np.sum(~nz_true))
        traj.f_add_result(f'posi{splitrat}.postsigdet.fp', 0)
        traj.f_add_result(f'posi{splitrat}.postsigdet.fn', np.sum(nz_true))

    toc = clk()
    traj.f_add_result(f'posi{splitrat}.runtime', toc - tic)


def data_splitting(traj, X, Y, beta, splitrat=.5):
    tic = clk()
    n = X.shape[0]
    n_train = int(np.floor(n*splitrat))

    indices = np.random.permutation(n)

    train_idx, test_idx = indices[:n_train], indices[n_train:]

    X_train = X[train_idx, :]
    Y_train = Y[train_idx]

    X_test = X[test_idx, :]
    Y_test = Y[test_idx]

    nonzero, conv, dispersion = grp_lasso_selection(X_train, Y_train, traj, randomize=False)
    traj.f_add_result(f'split{splitrat}.nonzero.mask', nonzero)
    traj.f_add_result(f'split{splitrat}.nonzero.nnz', nonzero.sum())

    nz_true = beta.astype(bool)
    traj.f_add_result(f'split{splitrat}.sigdet.tp', np.logical_and(nonzero, nz_true).sum())
    traj.f_add_result(f'split{splitrat}.sigdet.tn', np.logical_and(~nonzero, ~nz_true).sum())
    traj.f_add_result(f'split{splitrat}.sigdet.fp', np.logical_and(nonzero, ~nz_true).sum())
    traj.f_add_result(f'split{splitrat}.sigdet.fn', np.logical_and(~nonzero, nz_true).sum())

    if nonzero.sum() > 0:
        prior_var = 100 * dispersion

        mupos, sigmapos = compute_posterior_distribution(prior_var,
                                                         dispersion,
                                                         X_test[:, nonzero],
                                                         Y_test)

        beta_target = np.linalg.pinv(X_test[:, nonzero]).dot(X_test.dot(beta))

        traj.f_add_result(f'split{splitrat}.mse.target', compute_mse_target(mupos, beta_target))
        traj.f_add_result(f'split{splitrat}.mse.truth', compute_mse_truth(mupos, beta, nonzero))

        intervals = naive_confidence_intervals(np.diag(sigmapos), mupos)

        lci = intervals[:, 0]
        uci = intervals[:, 1]

        coverage = (lci < beta_target) * (uci > beta_target)
        length = uci - lci

        traj.f_add_result(f'split{splitrat}.componentwise.coverage', coverage)
        traj.f_add_result(f'split{splitrat}.componentwise.length', length)

        traj.f_add_result(f'split{splitrat}.mean.coverage', np.mean(coverage))
        traj.f_add_result(f'split{splitrat}.mean.length', np.mean(length))

        post_coverage = compute_coverage(lci, uci, beta, nonzero)
        traj.f_add_result(f'split{splitrat}.postcoverage', np.mean(post_coverage))


        tp, tn, fp, fn = compute_post_sigdet(lci, uci, beta, nonzero)
        traj.f_add_result(f'split{splitrat}.postsigdet.tp', tp)
        traj.f_add_result(f'split{splitrat}.postsigdet.tn', tn)
        traj.f_add_result(f'split{splitrat}.postsigdet.fp', fp)
        traj.f_add_result(f'split{splitrat}.postsigdet.fn', fn)
    else:
        traj.f_add_result(f'split{splitrat}.mse.target', np.nan)
        traj.f_add_result(f'split{splitrat}.mse.truth', compute_mse_truth(0, beta, nonzero))

        traj.f_add_result(f'split{splitrat}.componentwise.coverage', np.nan)
        traj.f_add_result(f'split{splitrat}.componentwise.length', np.nan)

        traj.f_add_result(f'split{splitrat}.mean.coverage', np.nan)
        traj.f_add_result(f'split{splitrat}.mean.length', np.nan)

        traj.f_add_result(f'split{splitrat}.postcoverage', 1 - np.mean(nz_true))


        traj.f_add_result(f'split{splitrat}.postsigdet.tp', 0)
        traj.f_add_result(f'split{splitrat}.postsigdet.tn', np.sum(~nz_true))
        traj.f_add_result(f'split{splitrat}.postsigdet.fp', 0)
        traj.f_add_result(f'split{splitrat}.postsigdet.fn', np.sum(nz_true))

    toc = clk()
    traj.f_add_result(f'split{splitrat}.runtime', toc - tic)

def compute_mse_target(betahat, beta):
    p = betahat.shape[0]
    mse = 1/p * np.sum((betahat - beta) ** 2)
    return mse


def compute_mse_truth(betahat, beta, nonzero):
    betahat_augment = np.zeros(beta.shape)
    betahat_augment[nonzero] = betahat
    mse = compute_mse_target(betahat_augment, beta)
    return mse


def compute_posterior_distribution(prior_var, dispersion, X, Y):
    p = X.shape[1]
    Q = X.T.dot(X)
    prior_prec = 1./prior_var * np.eye(p)
    sigmapos = np.linalg.inv(prior_prec + 1/dispersion * Q)
    mupos = 1./dispersion * sigmapos.dot(X.T).dot(Y)
    return mupos, sigmapos


def compute_coverage(lci, uci, beta, nonzero):
    """Compute coverage with respect to the original beta

    Construct "expanded" intervals by setting the interval to {0} for anything
    that wasn't selected, and then compute coverage with respect to the
    original beta. As a consequence, anything appropriately *not* selected
    counts as being covered.
    """
    lower = np.zeros(beta.shape)
    lower[nonzero] = lci
    upper = np.zeros(beta.shape)
    upper[nonzero] = uci

    coverage = (lower < beta) * (upper > beta)
    return coverage


def compute_post_sigdet(lci, uci, beta, nonzero):
    """Compute "Signal Detection" metrics based on intervals"""
    lower = np.zeros(beta.shape)
    lower[nonzero] = lci
    upper = np.zeros(beta.shape)
    upper[nonzero] = uci

    nz_true = beta.astype(bool)

    discoveries = np.logical_or(lower > 0, upper < 0)

    tp = np.logical_and(discoveries, nz_true).sum()
    tn = np.logical_and(~discoveries, ~nz_true).sum()
    fp = np.logical_and(discoveries, ~nz_true).sum()
    fn = np.logical_and(~discoveries, nz_true).sum()

    return tp, tn, fp, fn
