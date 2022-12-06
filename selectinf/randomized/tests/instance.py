import numpy as np

from scipy.stats import t as tdist

_cov_cache = {}

def _design(n, p, rho, equicorrelated):
    """
    Create an equicorrelated or AR(1) design.
    """
    if equicorrelated:
        X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
             np.sqrt(rho) * np.random.standard_normal(n)[:, None])
        def equi(rho, p):
            if ('equi', p, rho) not in _cov_cache:
                sigmaX = (1 - rho) * np.identity(p) + rho * np.ones((p, p))
                cholX = np.linalg.cholesky(sigmaX)
                _cov_cache[('equi', p, rho)] = sigmaX, cholX
            return _cov_cache[('equi', p, rho)]
        sigmaX, cholX = equi(rho=rho, p=p)
    else:
        def AR1(rho, p):
            if ('AR1', p, rho) not in _cov_cache:
                idx = np.arange(p)
                cov = rho ** np.abs(np.subtract.outer(idx, idx))
                _cov_cache[('AR1', p, rho)] = cov, np.linalg.cholesky(cov)
            cov, chol = _cov_cache[('AR1', p, rho)]
            return cov, chol
        sigmaX, cholX = AR1(rho=rho, p=p)
        X = np.random.standard_normal((n, p)).dot(cholX.T)
    return X, sigmaX, cholX

def gaussian_group_instance(n=100, p=200, sgroup=7, sigma=5, rho=0., signal=7,
                            random_signs=False, df=np.inf,
                            scale=True, center=True,
                            groups=np.arange(20).repeat(10),
                            equicorrelated=True):
    """A testing instance for the group LASSO.


    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\\lambda$ of around 13.5
    corresponds to the theoretical $E(\\|X^T\\epsilon\\|_{\\infty})$
    with $\\epsilon \\sim N(0, \\sigma^2 I)$.

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    sgroup : int or list
        True sparsity (number of active groups).
        If a list, which groups are active

    groups : array_like (1d, size == p)
        Assignment of features to (non-overlapping) groups

    sigma : float
        Noise level

    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(n).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive.

    df : int
        Degrees of freedom for noise (from T distribution).

    equicorrelated: bool
        If true, design in equi-correlated,
        Else design is AR.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.
    """
    from selectinf.tests.instance import _design
    X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

    if center:
        X -= X.mean(0)[None, :]

    beta = np.zeros(p)
    signal = np.atleast_1d(signal)

    group_labels = np.unique(groups)
    if isinstance(sgroup, list):
        group_active = sgroup
    else:
        group_active = np.random.choice(group_labels, sgroup, replace=False)

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active.sum())
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if scale:
        # ----SCALE----
        # scales X by sqrt(n) and sd
        # if we need original X, uncomment the following line
        # X_indi_raw = X_indi
        # ----SCALE----
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)
        sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    # ----SCALE----
    # uncomment line 158 if we need unscaled data
    # note that in the usage of the function,
    # we should now take "X, Y, beta, X_raw = inst(...)[:4]",
    # where inst = gaussian_group_instance
    # return X, Y, beta * sigma, X_raw, np.nonzero(active)[0], sigma, sigmaX
    # ----SCALE----
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma, sigmaX

def logistic_group_instance(n=100, p=200, sgroup=7,
                            ndiscrete=4, nlevels=None, sdiscrete=2,
                            rho=0.3, signal=7,
                            random_signs=False,
                            scale=True, center=True,
                            groups=np.arange(20).repeat(10),
                            equicorrelated=True):
    """A testing instance for the group LASSO.


    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\\lambda$ of around 13.5
    corresponds to the theoretical $E(\\|X^T\\epsilon\\|_{\\infty})$

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    sgroup : int or list
        True sparsity (number of active groups).
        If a list, which groups are active

    ndiscrete: int
        Among the active groups, how many of them correspond to a discrete variable

    nlevels: int
        How many levels of values does the discrete variables take?
        If the groups are uniformly of size k, then nlevels = k + 1

    groups : array_like (1d, size == p)
        Assignment of features to (non-overlapping) groups

    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(n).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive

    equicorrelated: bool
        If true, design in equi-correlated,
        Else design is AR.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.
    """
    all_discrete = False
    all_cts = False
    if ndiscrete == 0:
        all_cts = True
    elif ndiscrete * (nlevels-1) == p:
        all_discrete = True

    def gen_one_variable():
        probabilities = 1 / nlevels * np.ones(nlevels)
        sample = np.random.choice(np.arange(nlevels), n, p=probabilities)
        X = np.zeros((n, nlevels - 1))
        for i in np.arange(nlevels):
            if i != 0:
                X[:, i - 1] = (sample == i).astype(int)

        return X

    def gen_discrete_variables():
        X = None
        for i in np.arange(ndiscrete):
            if i == 0:
                X = gen_one_variable()
            else:
                X = np.concatenate((X,gen_one_variable()),axis=1)
        return X

    if not all_cts:
        X_indi = gen_discrete_variables()

        if scale:
            # ----SCALE----
            # scales X by sqrt(n) and sd
            # if we need original X, uncomment the following line
            # X_indi_raw = X_indi
            # ----SCALE----
            scaling = X_indi.std(0) * np.sqrt(n)
            X_indi /= scaling[None, :]

    beta = np.zeros(p)
    signal = np.atleast_1d(signal)

    group_labels = np.unique(groups)

    ## We mark the first `ndiscrete` groups to correspond to discrete r.v.s
    if isinstance(sgroup, list):
        group_active = sgroup
    else:
        if all_cts:
            group_active = np.random.choice(group_labels, sgroup, replace=False)
        elif all_discrete:
            group_active = np.random.choice(group_labels, sdiscrete, replace=False)
            #print("None null discrete variables:", np.sort(group_active))
        else:
            group_active = np.random.choice(np.arange(ndiscrete), sdiscrete, replace=False)
            #print("None null discrete variables:", np.sort(group_active))
            non_discrete_groups = np.setdiff1d(group_labels,np.arange(ndiscrete))
            group_active = np.append(group_active,
                                     np.random.choice(non_discrete_groups, sgroup-sdiscrete, replace=False))
    #print('true active groups:', np.sort(group_active))

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active.sum())
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if all_discrete:
        X = X_indi
        sigmaX = None
    elif all_cts:
        from selectinf.tests.instance import _design
        X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

        if center:
            X -= X.mean(0)[None, :]

        if scale:
            # ----SCALE----
            # scales X by sqrt(n) and sd
            # if we need original X, uncomment the following line
            # X_raw = X
            # ----SCALE----
            scaling = X.std(0) * np.sqrt(n)
            X /= scaling[None, :]
            beta *= np.sqrt(n)
            sigmaX = sigmaX / np.multiply.outer(scaling, scaling)
    else:
        from selectinf.tests.instance import _design
        X, sigmaX = _design(n, p - ndiscrete * (nlevels - 1),
                            rho, equicorrelated)[:2]

        if center:
            X -= X.mean(0)[None, :]

        if scale:
            # ----SCALE----
            # scales X by sqrt(n) and sd
            # if we need original X, uncomment the following line
            # X_raw = X
            # ----SCALE----
            scaling = X.std(0) * np.sqrt(n)
            X /= scaling[None, :]
            beta *= np.sqrt(n)
            sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

        X = np.concatenate((X_indi,X),axis=1)

    # ----SCALE----
    # raw data
    # uncomment if we need unscaled data
    # X_raw = np.concatenate((X_indi_raw,X_raw),axis=1)
    # ----SCALE----

    eta = linpred = np.dot(X, beta)
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = np.random.binomial(1, pi)

    # ----SCALE----
    # uncomment line 366 if we need unscaled data
    # note that in the usage of the function,
    # we should now take "X, Y, beta, X_raw = inst(...)[:4]",
    # where inst = logistic_group_instance
    # return X, Y, beta, X_raw, np.nonzero(active)[0], sigmaX
    # ----SCALE----
    return X, Y, beta, np.nonzero(active)[0], sigmaX

def poisson_group_instance(n=100, p=200, sgroup=7,
                           ndiscrete=4, nlevels=None, sdiscrete=2,
                           rho=0.3, signal=7,
                           random_signs=False,
                           scale=True, center=True,
                           groups=np.arange(20).repeat(10),
                           equicorrelated=True):
    """A testing instance for the Poisson group LASSO.


    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\\lambda$ of around 13.5
    corresponds to the theoretical $E(\\|X^T\\epsilon\\|_{\\infty})$

    Parameters
    ----------

    n : int
        Sample size

    p : int
        Number of features

    sgroup : int or list
        True sparsity (number of active groups).
        If a list, which groups are active

    ndiscrete: int
        Among the active groups, how many of them correspond to a discrete variable

    nlevels: int
        How many levels of values does the discrete variables take?
        If the groups are uniformly of size k, then nlevels = k + 1

    groups : array_like (1d, size == p)
        Assignment of features to (non-overlapping) groups

    rho : float
        Equicorrelation value (must be in interval [0,1])

    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
        Note: the size of signal is for a "normalized" design, where np.diag(X.T.dot(X)) == np.ones(n).
        If scale=False, this signal is divided by np.sqrt(n), otherwise it is unchanged.

    random_signs : bool
        If true, assign random signs to coefficients.
        Else they are all positive

    equicorrelated: bool
        If true, design in equi-correlated,
        Else design is AR.

    Returns
    -------

    X : np.float((n,p))
        Design matrix.

    y : np.float(n)
        Response vector.

    beta : np.float(p)
        True coefficients.

    active : np.int(s)
        Non-zero pattern.

    sigma : float
        Noise level.

    sigmaX : np.ndarray((p,p))
        Row covariance.
    """
    all_discrete = False
    all_cts = False
    if ndiscrete == 0:
        all_cts = True
    elif ndiscrete * (nlevels-1) == p:
        all_discrete = True

    def gen_one_variable():
        probabilities = 1 / nlevels * np.ones(nlevels)
        sample = np.random.choice(np.arange(nlevels), n, p=probabilities)
        X = np.zeros((n, nlevels - 1))
        for i in np.arange(nlevels):
            if i != 0:
                X[:, i - 1] = (sample == i).astype(int)

        return X

    def gen_discrete_variables():
        X = None
        for i in np.arange(ndiscrete):
            if i == 0:
                X = gen_one_variable()
            else:
                X = np.concatenate((X,gen_one_variable()),axis=1)
        return X

    if not all_cts:
        X_indi = gen_discrete_variables()

        if scale:
            scaling = X_indi.std(0) * np.sqrt(n)
            X_indi /= scaling[None, :]

    beta = np.zeros(p)
    signal = np.atleast_1d(signal)

    group_labels = np.unique(groups)

    ## We mark the first `ndiscrete` groups to correspond to discrete r.v.s
    if isinstance(sgroup, list):
        group_active = sgroup
    else:
        if all_cts:
            group_active = np.random.choice(group_labels, sgroup, replace=False)
        elif all_discrete:
            group_active = np.random.choice(group_labels, sdiscrete, replace=False)
            #print("None null discrete variables:", np.sort(group_active))
        else:
            group_active = np.random.choice(np.arange(ndiscrete), sdiscrete, replace=False)
            #print("None null discrete variables:", np.sort(group_active))
            non_discrete_groups = np.setdiff1d(group_labels,np.arange(ndiscrete))
            group_active = np.append(group_active,
                                     np.random.choice(non_discrete_groups, sgroup-sdiscrete, replace=False))
    #print('true active groups:', np.sort(group_active))

    active = np.isin(groups, group_active)

    if signal.shape == (1,):
        beta[active] = signal[0]
    else:
        beta[active] = np.linspace(signal[0], signal[1], active.sum())
    if random_signs:
        beta[active] *= (2 * np.random.binomial(1, 0.5, size=(active.sum(),)) - 1.)
    beta /= np.sqrt(n)

    if all_discrete:
        X = X_indi
        sigmaX = None
    elif all_cts:
        from selectinf.tests.instance import _design
        X, sigmaX = _design(n, p, rho, equicorrelated)[:2]

        if center:
            X -= X.mean(0)[None, :]

        if scale:
            scaling = X.std(0) * np.sqrt(n)
            X /= scaling[None, :]
            beta *= np.sqrt(n)
            sigmaX = sigmaX / np.multiply.outer(scaling, scaling)
    else:
        from selectinf.tests.instance import _design
        X, sigmaX = _design(n, p - ndiscrete * (nlevels - 1),
                            rho, equicorrelated)[:2]

        if center:
            X -= X.mean(0)[None, :]

        if scale:
            scaling = X.std(0) * np.sqrt(n)
            X /= scaling[None, :]
            beta *= np.sqrt(n)
            sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

        X = np.concatenate((X_indi,X),axis=1)

    eta = linpred = np.dot(X, beta)
    lambda_ = np.exp(eta)

    Y = np.random.poisson(lam=lambda_)

    return X, Y, beta, np.nonzero(active)[0], sigmaX
