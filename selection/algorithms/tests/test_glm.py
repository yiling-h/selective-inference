import numpy as np

from selection.algorithms.glm import (PH_lasso, Logistic_lasso, 
                                      Poisson_lasso, OLS_lasso)

def test_coxph(n=150, p=10):

    def instance():

        X = np.random.standard_normal((n, p))
        T = np.array(sorted(np.exp(np.random.standard_normal(X.shape[0]))))
        E = np.random.binomial(1, 0.7, size=(n,)).astype(np.bool)

        return X, T, E

    X, T, E = instance()
    lam = np.ones(p) * 7
    lam[0] = 0
    lam[3] = 0
    cox_lasso = PH_lasso(X, T, E, lam)
    cox_lasso.fit(solve_args={'tol':1.e-10, 'min_its':100})

    I = cox_lasso.intervals
    return [p for _, p in cox_lasso.active_pvalues]

def test_logistic(n=150, p=10):

    def instance():

        X = np.random.standard_normal((n, p))
        Y = np.random.binomial(1, 0.5, size=(n,))

        return X, Y

    X, Y = instance()
    lam = np.ones(p) * 7
    lam[0] = 0
    lam[3] = 0
    logit_lasso = Logistic_lasso(X, Y, lam)
    logit_lasso.fit(solve_args={'tol':1.e-10, 'min_its':100})

    I = logit_lasso.intervals
    return [p for _, p in logit_lasso.active_pvalues]

def test_poisson(n=150, p=10):

    def instance():

        X = np.random.standard_normal((n, p))
        Y = np.random.poisson(5, size=(n,))

        return X, Y

    X, Y = instance()
    lam = np.ones(p) * 7
    lam[0] = 0
    lam[3] = 0
    poisson_lasso = Poisson_lasso(X, Y, lam)
    poisson_lasso.fit(solve_args={'tol':1.e-10, 'min_its':100})

    I = poisson_lasso.intervals
    return [p for _, p in poisson_lasso.active_pvalues]

def test_gaussian(n=150, p=10):

    def instance():

        X = np.random.standard_normal((n, p))
        Y = np.random.poisson(5, size=(n,))

        return X, Y

    X, Y = instance()
    lam = np.ones(p) * 7
    lam[0] = 0
    lam[3] = 0

    ols_lasso = OLS_lasso(X, Y, lam)
    ols_lasso.fit(solve_args={'tol':1.e-10, 'min_its':100})

    I = ols_lasso.intervals
    return [p for _, p in ols_lasso.active_pvalues]

