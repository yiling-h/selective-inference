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
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma, sigmaX

def logistic_group_instance(n=100, p=200, sgroup=7, rho=0.3, signal=7,
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
        scaling = X.std(0) * np.sqrt(n)
        X /= scaling[None, :]
        beta *= np.sqrt(n)
        sigmaX = sigmaX / np.multiply.outer(scaling, scaling)

    eta = linpred = np.dot(X, beta)
    pi = np.exp(eta) / (1 + np.exp(eta))

    Y = np.random.binomial(1, pi)

    return X, Y, beta, np.nonzero(active)[0], sigmaX
