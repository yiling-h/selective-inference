import os, numpy as np
from selection.algorithms.lasso import lasso

if __name__ == "__main__":

    ###read input files
    path = '/Users/snigdhapanigrahi/Test_egenes/sim_Egene_data/'

    gene = str("ENSG00000187642.5")
    X = np.load(os.path.join(path + "X_" + gene) + ".npy")
    n, p = X.shape
    X -= X.mean(0)[None, :]
    X /= (X.std(0)[None, :] * np.sqrt(n))
    X_unpruned = X

    prototypes = np.loadtxt(os.path.join(path + "protoclust_" + gene) + ".txt", delimiter='\t')
    prototypes = np.unique(prototypes).astype(int)
    print("prototypes", prototypes.shape[0])
    X = X[:, prototypes]
    n, p = X.shape
    print("shape of X", n, p)

    lam_frac = 0.80
    lam = lam_frac * np.mean(np.fabs(np.dot(X.T, np.random.standard_normal((n, 2000)))).max(0))
    split_risk = 0.
    split_covered = 0.
    split_length = 0.
    count = 0
    ndraw = 1000

    for seed_n in range(ndraw):

        np.random.seed(seed_n)
        y = np.random.standard_normal(n)

        subsample_size = int(0.80 * n)


        sel_idx = np.zeros(n, np.bool)
        sel_idx[:subsample_size] = 1
        np.random.shuffle(sel_idx)

        L = lasso.gaussian(X[sel_idx,:], y[sel_idx], lam, sigma=1.)
        soln = L.fit()
        active = soln != 0
        nactive = active.sum()
        active_set = np.nonzero(active)[0]
        print("active set", active_set)

        X_inf = X[~sel_idx,:]
        y_inf = y[~sel_idx]
        true_mean = np.zeros(n-subsample_size)

        if nactive>0:
            projection_active = X_inf[:, active].dot(np.linalg.inv(X_inf[:, active].T.dot(X_inf[:, active])))
            true_val = projection_active.T.dot(true_mean)

            point_est = projection_active.T.dot(y_inf)
            sd = np.sqrt(np.linalg.inv(X_inf[:, active].T.dot(X_inf[:, active])).diagonal())
            split_intervals = np.vstack([point_est - 1.65 * sd, point_est + 1.65 * sd]).T
            split_covered += np.mean((split_intervals[:, 0] <= true_val)*(true_val <= split_intervals[:, 1]))
            split_length += np.mean((split_intervals[:, 1] - split_intervals[:, 0]))
            split_risk += np.mean(np.power(point_est - true_val, 2.))
        else:
            count += 1
        print("iteration completed", seed_n)


print("results", count, split_covered/(ndraw-count), split_risk/(ndraw-count), split_length/(ndraw-count))




