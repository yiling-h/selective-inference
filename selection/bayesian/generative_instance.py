import numpy as np

def generate_data(n, p, sigma=1., rho=0., scale =True, center=True):

    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        scalingX = (X.std(0)[None, :] * np.sqrt(n))
        X /= scalingX

    beta_true = np.zeros(p)
    u = np.random.uniform(0., 1., p)
    for i in range(p):
        if u[i] <= 0.90:
            beta_true[i] = np.random.laplace(loc=0., scale=0.5)
        else:
            beta_true[i] = np.random.laplace(loc=0., scale=2.5)

    beta = beta_true

    Y = (X.dot(beta) + np.random.standard_normal(n)) * sigma

    return X, Y, beta* sigma, sigma, scalingX


def generate_data_instance(n, p, sigma=1., rho=0., s= 30, scale=True, center=True):
    X = (np.sqrt(1 - rho) * np.random.standard_normal((n, p)) + np.sqrt(rho) * np.random.standard_normal(n)[:, None])

    if center:
        X -= X.mean(0)[None, :]
    if scale:
        scalingX = (X.std(0)[None, :] * np.sqrt(n))
        X /= scalingX

    beta_true = np.zeros(p)
    strong = []
    null = []
    u = np.random.uniform(0., 1., s)
    for i in range(s):
        if u[i] <= 0.80:
            null.append(np.random.laplace(loc=0., scale=0.5))
        else:
            strong.append(np.random.laplace(loc=0., scale=2.5))
    strong = np.asarray(strong)
    null = np.asarray(null)
    position = np.linspace(0, p-1, num=strong.shape[0], dtype=np.int)
    position_bool = np.zeros(p, np.bool)
    position_bool[position] = 1
    beta_true[position_bool] = strong

    vec_indx = np.arange(p)
    beta_true[np.random.choice(vec_indx[position_bool==0], null.shape[0])] = null
    Y = (X.dot(beta_true) + np.random.standard_normal(n)) * sigma

    return X, Y, beta_true * sigma, sigma, scalingX