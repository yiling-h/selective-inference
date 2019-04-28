import numpy as np

def generate_data(n, p, sigma=1., rho=0., scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        X /= (X.std(0)[None, :] * np.sqrt(n))

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.95:
            beta_true[i] = np.random.laplace(loc=0., scale=0.1)
        else:
            beta_true[i] = np.random.laplace(loc=0., scale=1.)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta* sigma, sigma
