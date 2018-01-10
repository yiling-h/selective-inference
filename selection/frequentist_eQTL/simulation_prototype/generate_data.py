import numpy as np
import os
from scipy.stats import t as tdist

def _design(n, p, rho, equicorrelated):
    """
    Create an equicorrelated or AR(1) design.
    """
    if equicorrelated:
        X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) +
             np.sqrt(rho) * np.random.standard_normal(n)[:, None])
    else:
        def AR1(rho, p):
            idx = np.arange(p)
            cov = rho ** np.abs(np.subtract.outer(idx, idx))
            return cov, np.linalg.cholesky(cov)

        sigmaX, cholX = AR1(rho=rho, p=p)
        X = np.random.standard_normal((n, p)).dot(cholX.T)
    return X

def gaussian_instance(n=100, p=200, s=7, sigma=5, rho=0., signal=7,
                      random_signs=False, df=np.inf,
                      scale=True, center=True,
                      equicorrelated=False):

    """
    A testing instance for the LASSO.
    If equicorrelated is True design is equi-correlated in the population,
    normalized to have columns of norm 1.
    If equicorrelated is False design is auto-regressive.
    For the default settings, a $\lambda$ of around 13.5
    corresponds to the theoretical $E(\|X^T\epsilon\|_{\infty})$
    with $\epsilon \sim N(0, \sigma^2 I)$.
    Parameters
    ----------
    n : int
        Sample size
    p : int
        Number of features
    s : int
        True sparsity
    sigma : float
        Noise level
    rho : float
        Equicorrelation value (must be in interval [0,1])
    signal : float or (float, float)
        Sizes for the coefficients. If a tuple -- then coefficients
        are equally spaced between these values using np.linspace.
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
    """

    X = _design(n, p, rho, equicorrelated)

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None,:] * np.sqrt(n))
    beta = np.zeros(p)
    signal = np.atleast_1d(signal)

    if signal.shape == (1,):
        beta[:s] = signal[0]
    else:
        beta[:s] = np.linspace(signal[0], signal[1], s)
    if random_signs:
        beta[:s] *= (2 * np.random.binomial(1, 0.5, size=(s,)) - 1.)
    np.random.shuffle(beta)

    active = np.zeros(p, np.bool)
    active[beta != 0] = True

    # noise model
    def _noise(n, df=np.inf):
        if df == np.inf:
            return np.random.standard_normal(n)
        else:
            sd_t = np.std(tdist.rvs(df, size=50000))
        return tdist.rvs(df, size=n) / sd_t

    Y = (X.dot(beta) + _noise(n, df)) * sigma
    return X, Y, beta * sigma, np.nonzero(active)[0], sigma

if __name__ == "__main__":

    outdir = "/Users/snigdhapanigrahi/selective-inference/selection/frequentist_eQTL/simulation_prototype/data_directory/"
    negenes = 500
    regime = np.random.choice(range(6), negenes, replace=True, p=[0.40, 0.30, 0.15, 0.08, 0.05, 0.02])
    signals = [0, 1, 2, 3, 5, 10]

    for j in range(negenes):

        nsignals = signals[regime[j]]
        print("iteration completed", j)
        X, y, beta, active, sigma= gaussian_instance(n=100, p=1000, s=nsignals, sigma=1., rho=0.2, signal=3.,
                                                     random_signs=False, df=np.inf,scale=True,
                                                     center=True, equicorrelated=False)

        out_X = os.path.join(outdir + "X_" + str(j) + ".npy")
        np.save(out_X, X)

        y = y.reshape((y.shape[0],))
        out_Y = os.path.join(outdir + "y_" + str(j) + ".npy")
        np.save(out_Y, y)

        beta = beta.reshape((beta.shape[0],))
        out_beta = os.path.join(outdir + "b_" + str(j) + ".npy")
        np.save(out_beta, beta)








