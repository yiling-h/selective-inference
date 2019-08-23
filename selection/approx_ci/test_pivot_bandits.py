import numpy as np
from selection.tests.instance import _design
from selection.randomized.randomization import randomization

def generate_data(beta,
                  X,
                  K_ch,
                  n,
                  sigma=1.):
    dimX = X.ndim
    if dimX == 2:
        _, d = X.shape
    elif dimX==1:
        d= X.shape[0]
    else:
        raise ValueError("invalid entry for X")

    p = beta.shape[0]
    if p-d>0:
        D = np.zeros((n, p-d))
        D[:, K_ch] = 1

    if dimX == 2:
        A = np.hstack((X, D))
    else:
        A = np.append(X, D)

    Y = A.dot(beta) + np.random.standard_normal(n)*sigma

    return A, Y, beta, sigma

def bandit(A_0, Y_0, d, beta, sigma=1., randomizer_scale=1.):

    p = beta.shape[0]
    K = p - d

    contexts = np.zeros((p, K))
    for j in range(K):
        e = np.zeros(K)
        e[j] = 1
        x = _design(1, d, rho=0., equicorrelated= False)[:1]
        contexts[:, j] = np.append(x, e)

    cov_0 = np.linalg.inv(A_0.T.dot(A_0))
    w, V = np.linalg.eig(cov_0)
    target_0 = cov_0.dot(A_0.T.dot(Y_0))
    B = sigma * V.T.dot(np.sqrt(np.diag(w)).dot(V))

    w = randomization.isotropic_gaussian((beta.shape[0],), randomizer_scale*sigma).sample()
    reg = contexts.T.dot(target_0 + B.dot(w))

    ch = np.argmax(reg)
    ch_bool = np.zeros(K, np.bool)
    ch_bool[ch] = 1
    nch = np.array([z for z in range(K) if ch_bool[z] == 0])

    target_linear = np.zeros((K - 1, beta.shape[0]))
    for t in range(K - 1):
        target_linear[t, :] = contexts[:, nch[t]] - contexts[:, ch]
    #opt_linear = target_linear.dot(B)
    #print("check constraints ", (target_linear.dot(target_0 + B.dot(w))<=0.).sum(), K-1)

    a_1, y_1, _, _ = generate_data(beta=beta,
                                   X = (contexts[:, ch])[:d],
                                   K_ch=ch,
                                   n=1,
                                   sigma=sigma)

    A_1 = np.vstack((A_0, a_1))
    Y_1 = np.append(Y_0, y_1)

    cov_1 = np.linalg.inv(A_1.T.dot(A_1))
    target_1 = cov_1.dot(A_1.T.dot(Y_1))
    sd = sigma * np.sqrt(np.diag(cov_1))
    lower = target_1 - (1.65 * sd)
    upper = target_1 + (1.65 * sd)

    return ((lower < beta)*(beta < upper)), A_1, Y_1, (target_1-beta)

def greedy_bandit(A_0, Y_0, d, beta, sigma=1., randomizer_scale=0.5):

    p = beta.shape[0]
    K = p - d

    contexts = np.zeros((p, K))
    for j in range(K):
        e = np.zeros(K)
        e[j] = 1
        x = _design(1, d, rho=0., equicorrelated= False)[:1]
        contexts[:, j] = np.append(x, e)

    cov_0 = np.linalg.inv(A_0.T.dot(A_0))
    w, V = np.linalg.eig(cov_0)
    target_0 = cov_0.dot(A_0.T.dot(Y_0))
    B = sigma * V.T.dot(np.sqrt(np.diag(w)).dot(V))

    w = randomization.isotropic_gaussian((beta.shape[0],), randomizer_scale*sigma).sample()
    reg = contexts.T.dot(target_0 + B.dot(w))

    order = np.argsort(reg)[::-1]
    ch = order[0]
    ch_bool = np.zeros(K, np.bool)
    ch_bool[ch] = 1
    nch = np.array([z for z in range(K) if ch_bool[z] == 0])

    target_linear = np.zeros((K - 1, beta.shape[0]))
    for t in range(K - 1):
        target_linear[t, :] = contexts[:, nch[t]] - contexts[:, ch]

    a_1, y_1, _, _ = generate_data(beta=beta,
                                   X = (contexts[:, ch])[:d],
                                   K_ch=ch,
                                   n=1,
                                   sigma=sigma)

    A_1 = np.vstack((A_0, a_1))
    Y_1 = np.append(Y_0, y_1)

    cht = order[1]
    cov_1 = np.linalg.inv(A_1.T.dot(A_1))
    target_1 = cov_1.dot(A_1.T.dot(Y_1))
    sd = sigma * np.sqrt(np.diag(cov_1))
    lower = target_1[d:] - (1.96 * sd[d:])
    upper = target_1[d:] + (1.96 * sd[d:])

    return ((lower < beta[d:])*(beta[d:] < upper)), A_1, Y_1, (target_1-beta)[d:]

def test(rho=0., equicorrelated=False, T=1000, sigma=1.):

    narms = 20
    beta = np.append(np.append(np.array([0., 0.]), 1.8*np.ones(narms/2)), 1.8*np.ones(narms/2))
    D_list = []
    Y_0 = []
    for k in range(narms):
        X_0, _ = _design(n=30, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
        D_0, y_0, _, _ = generate_data(beta,
                                       X_0,
                                       K_ch=k,
                                       n=30,
                                       sigma=sigma)

        D_list.append(D_0)
        Y_0.append(y_0)

    A_0 = np.vstack(D_list)
    Y_0 = np.asarray(Y_0).ravel()
    print("check shape ", A_0.shape, Y_0.shape)

    # X_0, _ = _design(n=3, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    # D_0, y_0, _, _ = generate_data(beta,
    #                                X_0,
    #                                K_ch=0,
    #                                n=3,
    #                                sigma=sigma)
    #
    # X_1, _ = _design(n=10, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    # D_1, y_1, _, _ = generate_data(beta,
    #                                X_1,
    #                                K_ch=1,
    #                                n=10,
    #                                sigma=sigma)
    #
    # X_2, _ = _design(n=10, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    # D_2, y_2, _, _ = generate_data(beta,
    #                                X_2,
    #                                K_ch=2,
    #                                n=10,
    #                                sigma=sigma)
    #
    # X_3, _ = _design(n=3, p=2, rho=rho, equicorrelated=equicorrelated)[:2]
    # D_3, y_3, _, _ = generate_data(beta,
    #                                X_3,
    #                                K_ch=3,
    #                                n=3,
    #                                sigma=sigma)

    # A_0 = np.vstack((np.vstack((D_0, D_1)), D_2))
    # A_0 = np.vstack((A_0, D_3))
    # Y_0 = np.append(np.append(y_0, y_1), y_2)
    # Y_0 = np.append(Y_0, y_3)

    cov = np.zeros((T, narms))
    bias = np.zeros((T, narms))
    for t in range(T):
        cov[t,:], A_1, Y_1, bias[t, :] = greedy_bandit(A_0, Y_0, d=2, beta=beta, sigma=sigma)
        A_0 = A_1
        Y_0 = Y_1

    return cov, bias

T = 70
narms = 20
cov = np.zeros((T, narms))
bias = np.zeros((T, narms))
for i in range(300):
    result = test(T=T)
    cov += result[0]
    bias += result[1]
    print("iteration comp ", i + 1)
print("coverage so far ", i + 1, (cov/float(i+1)).tolist(), np.min(cov)/float(i+1))
#(bias/float(i+1)).tolist())





